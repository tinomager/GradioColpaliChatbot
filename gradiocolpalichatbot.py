import gradio as gr
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor
from PIL import Image
from io import BytesIO
from colpali_engine.models.paligemma import ColPali
from colpali_engine.models.paligemma import ColPaliProcessor
from pdf2image import convert_from_path
from pypdf import PdfReader
import numpy as np
from openai import AzureOpenAI
import base64
from datetime import datetime
import glob
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchIndex
)
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv

load_dotenv()
SEARCH_KEY = os.getenv("SEARCH_KEY")
SEARCH_ENDPOINT = os.getenv("SEARCH_ENDPOINT")
INDEX_NAME = os.getenv("INDEX_NAME")
LOGGING = os.getenv("LOGGING").lower() == "true"

class AzureClient:
    def __init__(self):
        self._client = None

    def get_client(self):
        if self._client is None:
            self._client = AzureOpenAI(api_key=os.environ['AZURE_OPENAI_API_KEY'],
                                       azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
                                       api_version=os.environ['OPENAI_API_VERSION'])
        return self._client

class SearchClientWrapper:
    def __init__(self, endpoint, key, index_name):
        self.credential = AzureKeyCredential(key)
        self.search_client = SearchClient(endpoint, index_name=index_name, credential=self.credential)
        self.index_client = SearchIndexClient(endpoint=endpoint, credential=self.credential)

    def index_exists(self, index_name):
        return any(index.name == index_name for index in self.index_client.list_indexes())

    def create_or_update_index(self, index):
        return self.index_client.create_or_update_index(index)

    def upload_documents(self, documents):
        self.search_client.upload_documents(documents=documents)

    def search(self, vector_query):
        if vector_query is None:
            return list(self.search_client.search(search_text=None))
        return list(self.search_client.search(search_text=None, vector_queries=[vector_query]))

class PDFProcessor:
    def __init__(self):
        pass

    @staticmethod
    def get_pdf_images(pdf_path):
        reader = PdfReader(pdf_path)
        page_texts = [page.extract_text() for page in reader.pages]
        images = convert_from_path(pdf_path)
        assert len(images) == len(page_texts)
        return images, page_texts

    @staticmethod
    def read_pdfs(subfolder="docs"):
        file_names = glob.glob(os.path.join(subfolder, "**", "*.pdf"), recursive=True)
        return [{"title": os.path.basename(filename), "url": filename} for filename in file_names]

class ImageProcessor:
    def __init__(self):
        pass

    @staticmethod
    def scale_image(image, max_width):
        w, h = image.size
        if w > max_width:
            ratio = max_width / float(w)
            new_h = int(h * ratio)
            return image.resize((max_width, new_h))
        return image

    @staticmethod
    def get_base64_image(image, add_url_prefix=False):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return "data:image/png;base64," + img_str if add_url_prefix else img_str

class ModelWrapper:
    def __init__(self, model_name, device, dtype):
        self.model = ColPali.from_pretrained("vidore/colpaligemma-3b-pt-448-base", torch_dtype=dtype).eval()
        self.model.load_adapter(model_name)
        self.model = self.model.eval().to(device)
        self.processor = ColPaliProcessor.from_pretrained(model_name)

    def process_images(self, images):
        return self.processor.process_images(images)

    def get_embeddings(self, inputs):
        with torch.no_grad():
            return self.model(**inputs)

class PDFIndexer:
    def __init__(self, search_client, model_wrapper, image_processor):
        self.search_client = search_client
        self.model_wrapper = model_wrapper
        self.image_processor = image_processor

    def create_pdf_search_index_and_upload_documents(self, index_name):
        if self.search_client.index_exists(index_name):
            results = self.search_client.search(vector_query=None)
            self.log_query_results("Whole search index", results)
            return self.search_client.index_client.get_index(index_name)

        all_pdfs = PDFProcessor.read_pdfs()
        for pdf in all_pdfs:
            page_images, page_texts = PDFProcessor.get_pdf_images(pdf['url'])
            pdf['images'] = page_images
            pdf['texts'] = page_texts

            page_embeddings = []
            dataloader = DataLoader(
                pdf['images'],
                batch_size=2,
                shuffle=False,
                collate_fn=lambda x: self.model_wrapper.process_images(x),
            )
            for batch_doc in tqdm(dataloader):
                batch_doc = {k: v.to(self.model_wrapper.model.device) for k, v in batch_doc.items()}
                embeddings = self.model_wrapper.get_embeddings(batch_doc)
                mean_embedding = torch.mean(embeddings, dim=1).float().cpu().numpy()
                page_embeddings.extend(mean_embedding)
            pdf['embeddings'] = page_embeddings

        lst_feed = []
        for pdf in all_pdfs:
            url = pdf['url']
            title = pdf['title']
            for page_number, (page_text, embedding, image) in enumerate(zip(pdf['texts'], pdf['embeddings'], pdf['images'])):
                base_64_image = self.image_processor.get_base64_image(self.image_processor.scale_image(image, 640), add_url_prefix=False)
                page = {
                    "id": str(hash(url + str(page_number))),
                    "url": url,
                    "title": title,
                    "page_number": page_number,
                    "image": base_64_image,
                    "text": page_text,
                    "embedding": embedding.tolist()
                }
                lst_feed.append(page)

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="myHnsw",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "metric": "cosine"
                    }
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw",
                    vectorizer="myVectorizer"
                )
            ]
        )

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
            SimpleField(name="url", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="title", type=SearchFieldDataType.String, searchable=True, retrievable=True),
            SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            SimpleField(name="image", type=SearchFieldDataType.String, retrievable=True),
            SearchableField(name="text", type=SearchFieldDataType.String, searchable=True, retrievable=True),
            SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=128, vector_search_profile_name="myHnswProfile")
        ]

        index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
        result = self.search_client.create_or_update_index(index)
        self.search_client.upload_documents(documents=lst_feed)
        return result

    def log_query_results(self, query, response):
        if not LOGGING:
            return

        html_content = f"<h3>Query text: '{query}', top results:</h3>"
        for i, hit in enumerate(response):
            title = hit["title"]
            url = hit["url"]
            page = hit["page_number"]
            image = hit["image"]
            score = hit["@search.score"]

            html_content += f"<h4>PDF Result {i + 1}</h4>"
            html_content += f'<p><strong>Title:</strong> <a href="{url}">{title}</a>, page {page+1} with score {score:.2f}</p>'
            html_content += f'<img src="data:image/png;base64,{image}" style="max-width:100%;">'

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(f"logs/query_results_{timestamp}.html", "w") as f:
            f.write(html_content)

class Chatbot:
    def __init__(self, azure_client, search_client, model_wrapper, image_processor):
        self.azure_client = azure_client
        self.search_client = search_client
        self.model_wrapper = model_wrapper
        self.image_processor = image_processor

    def process_query(self, query):
        mock_image = Image.new('RGB', (224, 224), color='white')
        inputs = self.model_wrapper.processor(text=query, images=mock_image, return_tensors="pt")
        inputs = {k: v.to(self.model_wrapper.model.device) for k, v in inputs.items()}
        embeddings = self.model_wrapper.get_embeddings(inputs)
        return torch.mean(embeddings, dim=1).float().cpu().numpy().tolist()[0]

    def chat_and_update_images(self, message, history, image1, image2, image3):
        vector_query = VectorizedQuery(
            vector=self.process_query(message),
            k_nearest_neighbors=5,
            fields="embedding",
        )
        results = self.search_client.search(vector_query)
        self.log_query_results(message, results)

        message_object = [
            {
                "role": "system",
                "content": """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.You will be given a mixed of text, tables, and image(s) usually of charts or graphs."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": message},
                    *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x["image"]}', "detail": "low"}}, results),
                ],
            }
        ]

        response = self.azure_client.get_client().chat.completions.create(
            model="gpt-4o",
            messages=message_object,
            max_tokens=4096,
        )

        ai_response = response.choices[0].message.content
        history.append((message, ai_response))

        image_paths = [f"images/image{i+1}.png" for i in range(len(results))]
        for image in image_paths:
            if os.path.exists(image):
                os.remove(image)

        for i, result in enumerate(results):
            image_data = base64.b64decode(result["image"])
            with open(image_paths[i], "wb") as f:
                f.write(image_data)

        return history, image_paths[0], image_paths[1], image_paths[2]

    def log_query_results(self, query, response):
        if not LOGGING:
            return

        html_content = f"<h3>Query text: '{query}', top results:</h3>"
        for i, hit in enumerate(response):
            title = hit["title"]
            url = hit["url"]
            page = hit["page_number"]
            image = hit["image"]
            score = hit["@search.score"]

            html_content += f"<h4>PDF Result {i + 1}</h4>"
            html_content += f'<p><strong>Title:</strong> <a href="{url}">{title}</a>, page {page+1} with score {score:.2f}</p>'
            html_content += f'<img src="data:image/png;base64,{image}" style="max-width:100%;">'

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(f"logs/query_results_{timestamp}.html", "w") as f:
            f.write(html_content)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    azure_client = AzureClient()
    search_client = SearchClientWrapper(SEARCH_ENDPOINT, SEARCH_KEY, INDEX_NAME)
    model_wrapper = ModelWrapper("vidore/colpali-v1.2", device, dtype)
    image_processor = ImageProcessor()
    pdf_indexer = PDFIndexer(search_client, model_wrapper, image_processor)
    chatbot = Chatbot(azure_client, search_client, model_wrapper, image_processor)

    pdf_indexer.create_pdf_search_index_and_upload_documents(INDEX_NAME)

    with gr.Blocks() as demo:
        chatbot_ui = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot_ui])

        with gr.Row():
            image1 = gr.Image(interactive=False, show_download_button=True)
            image2 = gr.Image(interactive=False, show_download_button=True)
            image3 = gr.Image(interactive=False, show_download_button=True)

        msg.submit(chatbot.chat_and_update_images, [msg, chatbot_ui, image1, image2, image3], [chatbot_ui, image1, image2, image3])

    demo.launch()

if __name__ == "__main__":
    main()
