import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import json
import time
import csv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=api_key)

def prepare_data():
    # Read the first n rows from Excel files
    n = 2000  # Adjust this number as needed
    
    # Read Medicine data
    df_medicine = pd.read_excel('Medicine_description.xlsx', sheet_name='Sheet1', header=0, nrows=n)
    
    # Get unique values in the Reason column and create mapping
    reasons = df_medicine["Reason"].unique()
    reasons_dict = {reason: i for i, reason in enumerate(reasons)}
    
    # Format the prompts and completions
    training_data = []
    for _, row in df_medicine.iterrows():
        item = {
            "messages": [
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": f"Drug: {row['Drug_Name']}\nMalady:"},
                {"role": "assistant", "content": f"{reasons_dict[row['Reason']]}"}
            ]
        }
        training_data.append(item)
    
    # Save the mapping for later use
    with open('class_mapping.json', 'w') as f:
        json.dump(reasons_dict, f, indent=2)
    
    # Write the prepared data in JSONL format
    with open("training_data.jsonl", "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

def save_results_to_csv(events, filename='training_results.csv'):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['created_at', 'level', 'message', 'type'])
        for event in events:
            writer.writerow([
                event.created_at,
                event.level,
                event.message,
                event.type
            ])

def main():
    # Prepare the training data
    prepare_data()
    
    # Upload the training file
    with open("training_data.jsonl", "rb") as f:
        response = client.files.create(
            file=f,
            purpose='fine-tune'
        )
    
    file_id = response.id
    print(f"File uploaded with ID: {file_id}")
    
    # Create fine-tuning job
    response = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-3.5-turbo"
    )
    
    job_id = response.id
    print(f"Fine-tuning job created with ID: {job_id}")
    
    # Monitor the fine-tuning process
    while True:
        status_response = client.fine_tuning.jobs.retrieve(job_id)
        status = status_response.status
        print(f"Status: {status}")
        
        if status == "succeeded":
            model_id = status_response.fine_tuned_model
            print(f"Fine-tuned model ID: {model_id}")
            
            # Save model ID and job ID to .env
            with open(".env", "a") as f:
                f.write(f"\nFINE_TUNED_MODEL={model_id}")
                f.write(f"\nFINE_TUNE_JOB_ID={job_id}")
            
            # Save training results
            results = client.fine_tuning.jobs.list_events(job_id)
            save_results_to_csv(results)
            print("Training results saved to training_results.csv")
            break
            
        elif status == "failed":
            print(f"Job failed: {getattr(status_response, 'error', 'No error details available')}")
            break
            
        time.sleep(30)

if __name__ == "__main__":
    main() 