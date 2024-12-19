# Import necessary libraries
from taipy.gui import Gui
from logic import preprocessing_image, extract_unique_breeds, predict_image
import numpy as np

# Initialize variables
content = ""
image_path = "./img/placeholder_image.png"
unique_breeds = extract_unique_breeds("unique_breeds_csv.csv")
prob = 0
pred = ""

# Define the HTML template for the user interface
index = """
<|text-center|
<|{"./img/logo.png"}|image||width=25vw|>

<|{content}|file_selector|extensions=.jpg|>
Select an image of a dog from your file system

<|{pred}|>

<|{image_path}|image|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>
>
"""
        
def on_change(state, var_name, var_value):
    # Handle changes in the GUI elements, particularly when a new image or breed is selected
    if var_name == "content":
        top_pred, top_prob = predict_image(var_value, unique_breeds)
        state.prob = round(top_prob * 100)
        state.pred = "This is a \n\n" + top_pred
        state.image_path = var_value

# Create an instance of the GUI
app = Gui(page=index)

# Run the application with reloader enabled
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, use_reloader=True)