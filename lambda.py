import json
import boto3
from transformers import FlorenceModel, FlorenceProcessor
from PIL import Image
import torch
import io

# Load Florence-2 model during cold start
model = FlorenceModel.from_pretrained("microsoft/florence2")
processor = FlorenceProcessor.from_pretrained("microsoft/florence2")

def lambda_handler(event, context):
    # Parse S3 event and download the image
    s3 = boto3.client('s3')
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    response = s3.get_object(Bucket=bucket, Key=key)
    image = Image.open(io.BytesIO(response['Body'].read())).convert('RGB')
    
    # Preprocess image and make predictions
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    predicted_tags = processor.decode(logits_per_image)
    
    return {
        'statusCode': 200,
        'body': json.dumps({'tags': predicted_tags})
    }
