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

client = AzureOpenAI(api_key=os.environ['AZURE_OPENAI_API_KEY'],
                    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
                    api_version=os.environ['OPENAI_API_VERSION'])

credential = AzureKeyCredential(SEARCH_KEY)
search_client = SearchClient(
    SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=credential,
)

def scale_image(image, max_width):
    # Example scaling logic
    w, h = image.size
    if w > max_width:
        ratio = max_width / float(w)
        new_h = int(h * ratio)
        return image.resize((max_width, new_h))
    return image

def get_base64_image(image, add_url_prefix=False):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    if add_url_prefix:
        return "data:image/png;base64," + img_str
    return img_str

if torch.cuda.is_available():
    device = torch.device("cuda")
    if torch.cuda.is_bf16_supported():
        type = torch.bfloat16
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    type = torch.float32
else:
    device = torch.device("cpu")
    type = torch.float32

model_name = "vidore/colpali-v1.2"
model = ColPali.from_pretrained("vidore/colpaligemma-3b-pt-448-base", torch_dtype=type).eval()
model.load_adapter(model_name)
model = model.eval()
model.to(device)
professor = ColPaliProcessor.from_pretrained(model_name)

def get_pdf_images(pdf_path):
    reader = PdfReader(pdf_path)
    page_texts = []
    for page_number in range(len(reader.pages)):
        page = reader.pages[page_number]
        text = page.extract_text()
        page_texts.append(text)
    images = convert_from_path(pdf_path)
    assert len(images) == len(page_texts)
    return (images, page_texts)

def create_pdf_search_index_and_upload_documents(endpoint: str, key: str, index_name: str) -> SearchIndex:
    # Initialize the search index client
    credential = AzureKeyCredential(key)
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
    index_search_client = SearchClient(
        SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=credential,
    )

    index_exists = any(index.name == index_name for index in index_client.list_indexes())
    if index_exists:
        results = index_search_client.search(search_text=None, filter=None, top=5000)
        results = list(results)
        log_query_results("Whole search index", results)
        return index_client.get_index(index_name)
    
    all_pdfs = read_pdfs()
    
    for pdf in all_pdfs:
        page_images, page_texts = get_pdf_images(pdf['url'])
        pdf['images'] = page_images
        pdf['texts'] = page_texts

        for pdf in all_pdfs:
            page_embeddings = []
            dataloader = DataLoader(
                    pdf['images'],
                    batch_size=2,
                    shuffle=False,
                    collate_fn=lambda x: professor.process_images(x),
                )
            for batch_doc in tqdm(dataloader):
                with torch.no_grad():
                    batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
                    embeddings = model(**batch_doc)
                    mean_embedding = torch.mean(embeddings, dim=1).float().cpu().numpy()
                    #page_embeddings.extend(list(torch.unbind(embeddings.to("cpu"))))
                    page_embeddings.extend(mean_embedding)
            pdf['embeddings'] = page_embeddings

        lst_feed = []
        for pdf in all_pdfs:
            url = pdf['url']
            title = pdf['title']
            for page_number, (page_text, embedding, image) in enumerate(zip(pdf['texts'], pdf['embeddings'], pdf['images'])):
                base_64_image = get_base64_image(scale_image(image,640),add_url_prefix=False)   
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

    # Define vector search configuration
    vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="myHnsw",
                    parameters={
                        "m": 4,  # Default HNSW parameter
                        "efConstruction": 400,  # Default HNSW parameter
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

    # Define the fields
    fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True
            ),
            SimpleField(
                name="url",
                type=SearchFieldDataType.String,
                filterable=True
            ),
            SearchableField(
                name="title",
                type=SearchFieldDataType.String,
                searchable=True,
                retrievable=True
            ),
            SimpleField(
                name="page_number",
                type=SearchFieldDataType.Int32,
                filterable=True,
                sortable=True
            ),
            SimpleField(
                name="image",
                type=SearchFieldDataType.String,
                retrievable=True
            ),
            SearchableField(
                name="text",
                type=SearchFieldDataType.String,
                searchable=True,
                retrievable=True
            ),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=128,
                vector_search_profile_name="myHnswProfile"
            )
        ]

    # Create the index definition
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search
    )

    # Create the index in Azure Cognitive Search
    result = index_client.create_or_update_index(index)
    #add all documents to the index
    search_client = SearchClient(endpoint=endpoint, credential=credential, index_name=index_name)
    search_client.upload_documents(documents=lst_feed)
    return result



def read_pdfs():
    all_pdfs=[]
    subfolder = "docs"
    file_names = glob.glob(os.path.join(subfolder, "**", "*.pdf"), recursive=True)

    for filename in file_names:
        all_pdfs.append(
            {
                "title": os.path.basename(filename),
                "url": filename
            }
        )
    return all_pdfs


def process_query(query: str, processor: AutoProcessor, model: ColPali) -> np.ndarray:
    mock_image = Image.new('RGB', (224, 224), color='white')

    inputs = processor(text=query, images=mock_image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        embeddings = model(**inputs)

    return torch.mean(embeddings, dim=1).float().cpu().numpy().tolist()[0]



def log_query_results(query, response):
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
        html_content += (
            f'<img src="data:image/png;base64,{image}" style="max-width:100%;">'
        )

    #display(HTML(html_content))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"logs/query_results_{timestamp}.html", "w") as f:
        f.write(html_content)

def chat_and_update_images(message, history, image1, image2, image3):
    # Hier Ihre Chatbot-Logik implementieren
    vector_query = VectorizedQuery(
        vector=process_query(message, professor, model),
        k_nearest_neighbors=5,
        fields="embedding",
    )
    results = search_client.search(search_text=None, vector_queries=[vector_query])
    results = list(results)
    log_query_results(message, results)

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

    response = client.chat.completions.create(
    model="gpt-4o",
    messages=message_object,
    max_tokens=4096,
    )

    ai_response = response.choices[0].message.content
    history.append((message, ai_response))
    
    # Beispiel für Bildaktualisierung (ersetzen Sie dies durch Ihre eigene Logik)
    image_paths = [f"images/image{i+1}.png" for i in range(len(results))]

    # Remove the old images first
    for image in image_paths:
        if os.path.exists(image):
            os.remove(image)

    for i, result in enumerate(results):
        image_data = base64.b64decode(result["image"])
        with open(image_paths[i], "wb") as f:
            f.write(image_data)
    
    return history, image_paths[0], image_paths[1], image_paths[2]

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    with gr.Row():
        image1 = gr.Image(interactive=False, show_download_button=True)
        image2 = gr.Image(interactive=False, show_download_button=True)
        image3 = gr.Image(interactive=False, show_download_button=True)

    msg.submit(chat_and_update_images, 
               [msg, chatbot, image1, image2, image3], 
               [chatbot, image1, image2, image3])

create_pdf_search_index_and_upload_documents(SEARCH_ENDPOINT, SEARCH_KEY, INDEX_NAME)
demo.launch()
