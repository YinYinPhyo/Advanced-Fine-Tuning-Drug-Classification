import os
from openai import OpenAI
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
fine_tuned_model = os.getenv("FINE_TUNED_MODEL")

if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
if not fine_tuned_model:
    raise ValueError("Model ID not found. Please run training first.")

client = OpenAI(api_key=api_key)

def load_class_mapping():
    """Load the class mapping from json file"""
    try:
        with open('class_mapping.json', 'r') as f:
            class_map = json.load(f)
            # Invert the dictionary to map numbers to maladies
            return {str(v): k for k, v in class_map.items()}
    except FileNotFoundError:
        raise FileNotFoundError("class_mapping.json not found. Please run training first.")

def test_model(drug_names):
    """Test the model with a list of drug names"""
    class_map = load_class_mapping()
    
    for drug_name in drug_names:
        try:
            # Create the prompt in the same format as training
            messages = [
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": f"Drug: {drug_name}\nMalady:"}
            ]
            
            # Get completion from the model
            response = client.chat.completions.create(
                model=fine_tuned_model,
                messages=messages,
                max_tokens=1,
                temperature=1
            )
            
            # Extract the predicted class
            predicted_class = response.choices[0].message.content.strip()
            
            # Map the class number back to malady name
            if predicted_class in class_map:
                malady = class_map[predicted_class]
                print(f"Drug: {drug_name}")
                print(f"Predicted Malady: {malady}")
            else:
                print(f"Unknown class prediction for {drug_name}: {predicted_class}")
                
        except Exception as e:
            print(f"Error processing {drug_name}: {str(e)}")
        print("-" * 50)

def main():
    # Test cases - you can modify these
    test_drugs = [
        "A CN Gel(Topical) 20gmA CN Soap 75gm",  # Should be for Acne
        "Addnok Tablet 20'S",                     # Should be for ADHD
        "ABICET M Tablet 10's",                   # Should be for Allergies
        "What is 'A CN Gel(Topical)' used for?",  # Testing with question format
    ]
    
    print("Testing fine-tuned model...")
    print("=" * 50)
    test_model(test_drugs)

if __name__ == "__main__":
    main() 