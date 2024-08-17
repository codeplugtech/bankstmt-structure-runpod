from dotenv import load_dotenv
import boto3
import pandas as pd
import os
import numpy as np
import cv2
import runpod
from runpod.serverless.utils import rp_download
from pdf2image import convert_from_path
from main import table_extractor

load_dotenv()

# Initialize S3 client
s3_client = boto3.client('s3', region_name='ap-south-1')
s3_bucket_name = os.getenv('AWS_BUCKET_NAME')


def handler(job):
    job_input = job.get('input', {})

    if not job_input.get("file_path", False):
        return {
            "error": "Input is missing the 'file_path' key. Please include a file_path and retry your request."
        }

    user_id = job_input.get('user_id', None)
    doc_id = job_input.get('doc_id', 'output')
    api = job_input.get('api', None)
    page_num = int(job_input.get('page_num', 1)) if user_id is not None or api is not None else 1
    print(page_num)

    file_path = job_input.get("file_path")
    downloaded_file = rp_download.file(file_path)
    file_path = downloaded_file.get('file_path')
    filename = (os.path.basename(file_path).split('.')[0])
    if file_path.lower().endswith('.pdf'):
        images = convert_from_path(file_path, first_page=1, last_page=page_num)
        pages = min(page_num, len(images)) if page_num is not None else len(images)
    else:
        images = [cv2.imread(file_path)]
        pages = 1
    output_excel_path = f"{filename}.xlsx"
    writer = pd.ExcelWriter(output_excel_path, engine='xlsxwriter')
    html = ""  # Initialize html here
    for page_number in range(pages):
        image = images[page_number]
        if not isinstance(image, np.ndarray):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        detection_result = table_extractor.table_detection(image)
        crop_images = table_extractor.crop_image(image, detection_result)
        for table_number, crop_image in enumerate(crop_images):
            words = table_extractor.ocr(crop_image)
            structure_result = table_extractor.table_structure(crop_image)
            table_structures, cells, confidence_score = table_extractor.convert_structure(words, crop_image,
                                                                                          structure_result)
            data_rows = table_extractor.visualize_cells(crop_image, table_structures, cells)
            sheet_name = f'Page_{page_number + 1}_Table_{table_number + 1}'
            df = pd.DataFrame(data_rows)
            df = df.rename(columns=df.iloc[0]).drop(df.index[0])
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            html = table_extractor.cells_to_html(cells)
    if api is None:
        if user_id is None:
            return {"refresh_worker": False, "job_results": 'html', "html_output": html}

    writer.close()
    if api == "True":
        s3_key = f"Api/{doc_id}.xlsx"
        s3_client.upload_file(output_excel_path, s3_bucket_name, f'{s3_key}')
    else:
        s3_key = f"excel/{doc_id}.xlsx"
        s3_client.upload_file(output_excel_path, s3_bucket_name, f'{user_id}/{s3_key}')
    try:
        print(f"file uploaded successfully")
    except Exception as e:
        return {"error": f"Error uploading to S3: {str(e)}"}

    return {"refresh_worker": False, "job_results": {"user_id": f"{user_id}", "doc_id": f"{doc_id}"}}


runpod.serverless.start({"handler": handler})

