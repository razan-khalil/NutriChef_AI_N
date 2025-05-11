import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from openai import OpenAI


class IngredientDetector:
    def __init__(self, device=None):
        """
        Initializes the BLIP model for ingredient detection from images.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

        # 🟰 Add OpenAI client inside the class
        self.client = OpenAI()

    def detect_ingredients(self, image):
        """
        Given a PIL image, returns a list of detected ingredients and the caption.
        """
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        ingredients = self.extract_keywords(caption)
        return ingredients, caption

    def extract_keywords(self, caption):
        """
         Use OpenAI GPT to smartly extract likely food ingredients from a scene description.
        """
    
        prompt = f"""
You are a food expert assistant.

Given the following description of a fridge or food scene:
"{caption}"

Guess the common food ingredients that are likely present, even if they are not explicitly mentioned.

Rules:
- Only list real food ingredients (e.g., milk, cheese, lettuce, apple).
- Ignore mentions of 'food', 'refrigerator', 'kitchen', 'it', or any general words.
- Make an educated guess based on the description context.
- Return your answer strictly as a Python list, e.g., ["milk", "lettuce", "cheese"].
- Do NOT include anything that isn't food.

If you are unsure, suggest typical fridge items.

Respond only with the Python list.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a food ingredient extraction assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )

            extracted = response.choices[0].message.content
            ingredients_list = eval(extracted)

            return ingredients_list

        except Exception as e:
            print(f"🔥 Error during ingredient extraction: {e}")
            return ["ingredient detection failed"]
