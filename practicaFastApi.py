from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
from PIL import Image

app = FastAPI()



# pipeline para clasificar imagenes, utilizamos un modelo de HF y luego mediante fastapi solictamos una imagen para hacer la prediccion
# modelo utilizado: https://huggingface.co/google/vit-base-patch16-224
clasificador = pipeline("image-classification", model="google/vit-base-patch16-224")

@app.post("/clasificadorImagenes")
async def clasificador_imagenes(file: UploadFile = File(...)):
    # leemos la imagen 
    image = Image.open(file.file)

    # clasificamos la imagen segun el modelo 
    return clasificador(image)

# pipeline para traduccion de texto de ingles a español 
# modelo utilizado: https://huggingface.co/Helsinki-NLP/opus-mt-en-es
traduccion = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")

@app.get("/traduccion")
def traduccion_texto(text:str):
    return traduccion(text)[0]['translation_text']


# pipeline para un analisis de sentimiento en castellano, este modelo soporta varios lenguajes y califica del 1 al 5 
# modelo utilizado: https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment
sentimiento = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

@app.get('/analisisSentimiento')
def analisis(text:str):
    return sentimiento(text)


# pipeline para identificar un tipo de coche
# modelo utilizado: https://huggingface.co/rok1958/Type_of_car
tipoCoche = pipeline("image-classification", model="rok1958/Type_of_car")

@app.post("/clasificarCoches")
async def clasificar_coches(file: UploadFile = File(...)):
    # leemos la imagen 
    image = Image.open(file.file)

    # clasificamos la imagen segun el modelo 
    return tipoCoche(image)



# pipeline para generar texto a raiz de una imagen
# modelo utilizado: https://huggingface.co/microsoft/git-base
imagenToTexto = pipeline("image-to-text", model="microsoft/git-base")

@app.post("/imagenToTexto")
async def imagen_texto(file: UploadFile = File(...)):
    # leemos la imagen 
    image = Image.open(file.file)

    # clasificamos la imagen segun el modelo 
    textoDescritoIngles =  imagenToTexto(image)[0]['generated_text']

    # utilizamos el modelo de traduccion para traduccir la descripcion 
    return traduccion(textoDescritoIngles)[0]['translation_text']