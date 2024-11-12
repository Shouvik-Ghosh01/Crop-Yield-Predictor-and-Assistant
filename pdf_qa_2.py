import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq
from datasets import Dataset
import os

# Set your Hugging Face API token as an environment variable
os.environ["HF_API_TOKEN"] = "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"


# Step 1: Predefined Question-Answer Pairs for both Wheat and Potato
def get_qa_pairs():
    qa_pairs = [
        # Wheat Crop Health and Disease Detection
        {"question": "What are the early signs of disease in wheat that I should watch for?",
         "answer": "Yellowing or browning of leaves, Leaf spots and blights, Stunted growth, Head blight and grain discoloration"},
        {"question": "What are common diseases affecting wheat in this region?",
         "answer": "Rust (leaf, stem, and stripe rust), Powdery mildew, Septoria leaf blotch, Fusarium head blight (head scab)"},
        {"question": "What factors increase the risk of disease in wheat?",
         "answer": "High humidity and warm temperatures, Poor drainage, Nutrient deficiencies, Overcrowding, Insect pests"},
        {"question": "Are there any soil-borne diseases that I should test for before planting wheat?",
         "answer": "Take-all, Fusarium root rot, Rhizoctonia root rot"},
        {"question": "What are the symptoms of fungal infections on wheat?",
         "answer": "White or gray powdery growth on leaves, Brown or black spots on leaves, Root rot and wilting, Head blight and grain discoloration"},
        {"question": "How does temperature and humidity affect disease spread in wheat?",
         "answer": "High humidity and warm temperatures favor fungal diseases. Dry conditions can stress plants and make them more susceptible to diseases"},
        {"question": "How frequently should I inspect wheat for signs of disease?",
         "answer": "Regular inspection, especially during flowering and grain filling stages"},
        {"question": "What are the most effective methods for detecting viral infections in wheat?",
         "answer": "Visual symptoms like leaf discoloration, stunting, and reduced yield. Laboratory tests for confirmation"},

        # Potato Crop Health and Disease Detection
        {"question": "What are the early signs of disease in potato that I should watch for?",
         "answer": "Yellowing or browning of leaves, Stunted growth, Tuber rot and discoloration, Late blight symptoms (dark, water-soaked lesions on leaves and stems)"},
        {"question": "What are common diseases affecting potato in this region?",
         "answer": "Late blight, Early blight, Potato scab, Rhizoctonia solani"},
        {"question": "What factors increase the risk of disease in potato?",
         "answer": "High humidity and warm temperatures, Poor drainage, Nutrient deficiencies, Overcrowding, Insect pests"},
        {"question": "Are there any soil-borne diseases that I should test for before planting potato?",
         "answer": "Rhizoctonia solani, Verticillium wilt, Fusarium wilt"},
        {"question": "What are the symptoms of fungal infections on potato?",
         "answer": "White or gray powdery growth on leaves, Brown or black spots on leaves, Root rot and wilting, Tuber rot and discoloration"},
        {"question": "How does temperature and humidity affect disease spread in potato?",
         "answer": "High humidity and warm temperatures favor fungal diseases. Dry conditions can stress plants and make them more susceptible to diseases"},
        {"question": "How frequently should I inspect potato for signs of disease?",
         "answer": "Regular inspection, especially during tuber formation and harvesting stages"},
        {"question": "What are the most effective methods for detecting viral infections in potato?",
         "answer": "Visual symptoms like mosaic patterns, leaf curling, and stunting. Laboratory tests for confirmation"},

        # Pest Management and Control
        {"question": "What pests commonly affect potato and how can I identify them?",
         "answer": "Colorado potato beetle, Aphids, White flies, Leaf hoppers"},
        {"question": "What are the signs of pest infestation in potato (e.g., leaf discoloration, holes)?",
         "answer": "Holes in leaves, Discolored or distorted leaves, Stunted growth, Reduced tuber yield"},
        {"question": "How does the use of organic pesticides impact potato compared to synthetic options?",
         "answer": "Organic pesticides have fewer environmental impacts but may be less effective. Synthetic pesticides can be more effective but can harm beneficial insects and the environment."},
        {"question": "Are there any beneficial insects or predators that help control pests for potato?",
         "answer": "Ladybugs, lacewings, and parasitic wasps can help control pests"},
        {"question": "What are the best practices for crop rotation to reduce pest infestations in potato?",
         "answer": "Rotate potatoes with non-host crops to reduce pest and disease pressure"},
        {"question": "How often should I apply pest control measures to ensure potato stays healthy?",
         "answer": "Use insecticides and miticides as needed, following label instructions"},
        {"question": "What are the most effective natural repellents for pests that target potato?",
         "answer": "Neem oil and other botanical insecticides can be effective"},
        {"question": "How can I monitor pest populations to prevent infestations in potato?",
         "answer": "Use traps and visual inspection to monitor pest populations"},

        # Preventive Measures and Best Practices
        {"question": "What are the best preventive measures to avoid disease in potato?",
         "answer": "Remove crop residues to reduce disease inoculum. Practice crop rotation. Use disease-free seed potatoes."},
        {"question": "How does proper irrigation impact the susceptibility of potato to disease?",
         "answer": "Avoid overwatering or underwatering. Use drip irrigation to minimize leaf wetness."},
        {"question": "What are the recommended soil treatments before planting potato to prevent disease?",
         "answer": "Soil solarization can help reduce soil-borne pathogens. Add organic matter to improve soil health."},
        {"question": "How can I improve soil health to make potato more resistant to disease and pests?",
         "answer": "Maintain soil fertility through proper fertilization and organic matter addition."},
        {"question": "What are common nutrient deficiencies in potato and how can I prevent them?",
         "answer": "Monitor nutrient levels and apply fertilizers as needed."},
        {"question": "How can intercropping help reduce pest and disease pressure on potato?",
         "answer": "Intercropping can help reduce pest and disease pressure."},
        {"question": "What are effective mulching techniques for preventing disease in potato?",
         "answer": "Mulching can help conserve moisture and suppress weeds."},
        {"question": "What are the early signs of disease in rice that I should watch for?",
         "answer": "Early signs of rice diseases include discolored or spotted leaves, stunted growth, and unusual plant appearance."},
        {"question": "What are common diseases affecting rice in this region?",
         "answer": "Common rice diseases in many regions include blast, sheath blight, bacterial leaf blight, and tungro virus."},
        {"question": "What factors increase the risk of disease in rice?",
         "answer": "Factors increasing disease risk in rice include high humidity, excessive nitrogen fertilization, poor drainage, and monocropping."},
        {"question": "Are there any soil-borne diseases that I should test for before planting rice?",
         "answer": "Yes, soil-borne diseases like sheath blight and bacterial blight can affect rice. Consider soil testing before planting."},

        {"question": "What are the early signs of disease in sugarcane that I should watch for?",
         "answer": "Yellowing or browning of leaves, red rot lesions on stems, stunted growth, and reduced sugar content."},
        {"question": "What are common diseases affecting sugarcane in this region?",
         "answer": "Red rot, smut, rust, and leaf blight are common sugarcane diseases."},
        {"question": "What factors increase the risk of disease in sugarcane?",
         "answer": "High humidity, poor drainage, nutrient deficiencies, overcrowding, and insect pests."},
        {"question": "Are there any soil-borne diseases that I should test for before planting sugarcane?",
         "answer": "Red rot and root rot are soil-borne diseases that may affect sugarcane."},

        {"question": "What are the early signs of disease in bajra that I should watch for?",
         "answer": "Yellowing or browning of leaves, leaf spots, stunted growth, head blight, and grain discoloration."},
        {"question": "What are common diseases affecting bajra in this region?",
         "answer": "Downy mildew, powdery mildew, leaf blight, and ergot are common in bajra."},
        {"question": "What factors increase the risk of disease in bajra?",
         "answer": "High humidity, poor drainage, nutrient deficiencies, overcrowding, and insect pests increase disease risk."},
        {"question": "Are there any soil-borne diseases that I should test for before planting bajra?",
         "answer": "Root rot caused by fungi like Rhizoctonia and Fusarium can affect bajra."},

        {"question": "What are the early signs of disease in black pepper that I should watch for?",
         "answer": "Yellowing or browning of leaves, leaf spots, dieback of branches, and root rot."},
        {"question": "What are common diseases affecting black pepper in this region?",
         "answer": "Phytophthora leaf blight, anthracnose, root (Koleroga) disease, and dieback are common diseases."},
        {"question": "What factors increase the risk of disease in black pepper?",
         "answer": "High humidity, poor drainage, nutrient deficiencies, and overcrowding."},
        {"question": "Are there any soil-borne diseases that I should test for before planting black pepper?",
         "answer": "Root (Koleroga) disease is a soil-borne disease that can affect black pepper."},

        {"question": "What are the early signs of disease in cardamom that I should watch for?",
         "answer": "Yellowing or browning of leaves, leaf spots, stunted growth, fruit rot, and premature fruit drop."},
        {"question": "What are common diseases affecting cardamom in this region?",
         "answer": "Leaf blight, anthracnose, root rot, and fruit rot are common cardamom diseases."},
        {"question": "What factors increase the risk of disease in cardamom?",
         "answer": "High humidity, poor drainage, nutrient deficiencies, and overcrowding increase disease risk."},
        {"question": "Are there any soil-borne diseases that I should test for before planting cardamom?",
         "answer": "Root rot caused by fungi like Rhizoctonia and Fusarium can affect cardamom."},

        {"question": "What are the early signs of disease in coriander that I should watch for?",
         "answer": "Yellowing or browning of leaves, leaf spots, stunted growth, root rot, and stem canker."},
        {"question": "What are common diseases affecting coriander in this region?",
         "answer": "Alternaria leaf blight, powdery mildew, and root rot."},
        {"question": "What factors increase the risk of disease in coriander?",
         "answer": "High humidity, poor drainage, nutrient deficiencies, overcrowding, and insect pests."},
        {"question": "Are there any soil-borne diseases that I should test for before planting coriander?",
         "answer": "Root rot caused by fungi like Rhizoctonia and Fusarium."},

        {"question": "What are the early signs of disease in garlic that I should watch for?",
         "answer": "Yellowing or browning of leaves, leaf spots, stunted growth, bulb rot, and discoloration."},
        {"question": "What are common diseases affecting garlic in this region?",
         "answer": "White rot, neck rot, leaf blight, and rust are common garlic diseases."},
        {"question": "What factors increase the risk of disease in garlic?",
         "answer": "High humidity, poor drainage, nutrient deficiencies, overcrowding, and insect pests increase risk."},
        {"question": "Are there any soil-borne diseases that I should test for before planting garlic?",
         "answer": "White rot and pink rot."},
        {"question": "What are the early signs of disease in rice that I should watch for?",
         "answer": "Early signs of rice diseases include discolored or spotted leaves, stunted growth, and unusual plant appearance."},
        {"question": "What are common diseases affecting rice in this region?",
         "answer": "Common rice diseases in many regions include blast, sheath blight, bacterial leaf blight, and tungro virus."},
        {"question": "What factors increase the risk of disease in rice?",
         "answer": "Factors increasing disease risk in rice include high humidity, excessive nitrogen fertilization, poor drainage, and monocropping."},
        {"question": "Are there any soil-borne diseases that I should test for before planting rice?",
         "answer": "Yes, soil-borne diseases like sheath blight and bacterial blight can affect rice. Consider soil testing before planting."},

        {"question": "What are the early signs of disease in sugarcane that I should watch for?",
         "answer": "Yellowing or browning of leaves, red rot lesions on stems, stunted growth, and reduced sugar content."},
        {"question": "What are common diseases affecting sugarcane in this region?",
         "answer": "Red rot, smut, rust, and leaf blight are common sugarcane diseases."},
        {"question": "What factors increase the risk of disease in sugarcane?",
         "answer": "High humidity, poor drainage, nutrient deficiencies, overcrowding, and insect pests."},
        {"question": "Are there any soil-borne diseases that I should test for before planting sugarcane?",
         "answer": "Red rot and root rot are soil-borne diseases that may affect sugarcane."},

        {"question": "What are the early signs of disease in bajra that I should watch for?",
         "answer": "Yellowing or browning of leaves, leaf spots, stunted growth, head blight, and grain discoloration."},
        {"question": "What are common diseases affecting bajra in this region?",
         "answer": "Downy mildew, powdery mildew, leaf blight, and ergot are common in bajra."},
        {"question": "What factors increase the risk of disease in bajra?",
         "answer": "High humidity, poor drainage, nutrient deficiencies, overcrowding, and insect pests increase disease risk."},
        {"question": "Are there any soil-borne diseases that I should test for before planting bajra?",
         "answer": "Root rot caused by fungi like Rhizoctonia and Fusarium can affect bajra."},

        {"question": "What are the early signs of disease in black pepper that I should watch for?",
         "answer": "Yellowing or browning of leaves, leaf spots, dieback of branches, and root rot."},
        {"question": "What are common diseases affecting black pepper in this region?",
         "answer": "Phytophthora leaf blight, anthracnose, root (Koleroga) disease, and dieback are common diseases."},
        {"question": "What factors increase the risk of disease in black pepper?",
         "answer": "High humidity, poor drainage, nutrient deficiencies, and overcrowding."},
        {"question": "Are there any soil-borne diseases that I should test for before planting black pepper?",
         "answer": "Root (Koleroga) disease is a soil-borne disease that can affect black pepper."},

        {"question": "What are the early signs of disease in cardamom that I should watch for?",
         "answer": "Yellowing or browning of leaves, leaf spots, stunted growth, fruit rot, and premature fruit drop."},
        {"question": "What are common diseases affecting cardamom in this region?",
         "answer": "Leaf blight, anthracnose, root rot, and fruit rot are common cardamom diseases."},
        {"question": "What factors increase the risk of disease in cardamom?",
         "answer": "High humidity, poor drainage, nutrient deficiencies, and overcrowding increase disease risk."},
        {"question": "Are there any soil-borne diseases that I should test for before planting cardamom?",
         "answer": "Root rot caused by fungi like Rhizoctonia and Fusarium can affect cardamom."},

        {"question": "What are the early signs of disease in coriander that I should watch for?",
         "answer": "Yellowing or browning of leaves, leaf spots, stunted growth, root rot, and stem canker."},
        {"question": "What are common diseases affecting coriander in this region?",
         "answer": "Alternaria leaf blight, powdery mildew, and root rot."},
        {"question": "What factors increase the risk of disease in coriander?",
         "answer": "High humidity, poor drainage, nutrient deficiencies, overcrowding, and insect pests."},
        {"question": "Are there any soil-borne diseases that I should test for before planting coriander?",
         "answer": "Root rot caused by fungi like Rhizoctonia and Fusarium."},

        {"question": "What are the early signs of disease in garlic that I should watch for?",
         "answer": "Yellowing or browning of leaves, leaf spots, stunted growth, bulb rot, and discoloration."},
        {"question": "What are common diseases affecting garlic in this region?",
         "answer": "White rot, neck rot, leaf blight, and rust are common garlic diseases."},
        {"question": "What factors increase the risk of disease in garlic?",
         "answer": "High humidity, poor drainage, nutrient deficiencies, overcrowding, and insect pests increase risk."},
        {"question": "Are there any soil-borne diseases that I should test for before planting garlic?",
         "answer": "White rot and pink rot."},

        {"question": "What are the early signs of disease in ginger that I should watch for?",
         "answer": "Yellowing or browning of leaves, leaf spots, stunted growth, rhizome rot, and discoloration."},
        {"question": "What are common diseases affecting ginger in this region?",
         "answer": "Rhizome rot, leaf blight, and virus diseases such as mosaic and yellow vein mosaic."},
        {"question": "What factors increase the risk of disease in ginger?",
         "answer": "High humidity, poor drainage, nutrient deficiencies, overcrowding, and insect pests increase disease risk."},
        {"question": "Are there any soil-borne diseases that I should test for before planting ginger?",
         "answer": "Root rot caused by fungi like Rhizoctonia and Fusarium."},

        {"question": "What are the early signs of disease in jowar that I should watch for?",
         "answer": "Yellowing or browning of leaves, leaf spots, stunted growth, head blight, and grain discoloration."},
        {"question": "What are common diseases affecting jowar in this region?",
         "answer": "Downy mildew, powdery mildew, leaf blight, and ergot are common in jowar."},
        {"question": "What factors increase the risk of disease in jowar?",
         "answer": "High humidity, poor drainage, nutrient deficiencies, overcrowding, and insect pests increase disease risk."},
        {"question": "Are there any soil-borne diseases that I should test for before planting jowar?",
         "answer": "Root rot caused by fungi like Rhizoctonia and Fusarium."},

        {"question": "What are the early signs of disease in ragi that I should watch for?",
         "answer": "Yellowing or browning of leaves, leaf spots, stunted growth, head blight, and grain discoloration."},
        {"question": "What are common diseases affecting ragi in this region?",
         "answer": "Powdery mildew, rust, leaf blight, and ergot are common diseases in ragi."},
        {"question": "What factors increase the risk of disease in ragi?",
         "answer": "High humidity, poor drainage, nutrient deficiencies, overcrowding, and insect pests increase disease risk."},
        {"question": "Are there any soil-borne diseases that I should test for before planting ragi?",
         "answer": "Root rot caused by fungi like Rhizoctonia and Fusarium."},

        {"question": "What are the early signs of disease in cashewnut that I should watch for?",
         "answer": "Yellowing or browning of leaves, leaf spots, dieback of branches, and root rot."},
        {"question": "What are common diseases affecting cashewnut in this region?",
         "answer": "Leaf blight, anthracnose, root rot, and dieback are common diseases."},
        {"question": "What factors increase the risk of disease in cashewnut?",
         "answer": "High humidity, poor drainage, nutrient deficiencies, overcrowding increase disease risk."},
        {"question": "Are there any soil-borne diseases that I should test for before planting cashewnut?",
         "answer": "Root rot caused by fungi like Rhizoctonia and Fusarium."},

        {"question": "What are the early signs of disease in banana that I should watch for?",
         "answer": "Yellowing or browning of leaves, leaf spots, stunted growth, fruit rot, and discoloration."},
        {"question": "What are common diseases affecting banana in this region?",
         "answer": "Panama disease, Sigatoka leaf spot, bunchy top disease, and black Sigatoka."},
        {"question": "What factors increase the risk of disease in banana?",
         "answer": "High humidity, poor drainage, nutrient deficiencies, overcrowding, and insect pests increase disease risk."},
        {"question": "How does planting season affect the likelihood of disease in potato?",
         "answer": "Plant at the recommended time to avoid stress from extreme weather conditions."}
    ]
    return qa_pairs


# Step 2: Load and Tokenize Dataset
def create_dataset(qa_pairs, tokenizer, max_length=512):
    inputs, targets = [], []
    for pair in qa_pairs:
        inputs.append("question: " + pair["question"])
        targets.append(pair["answer"])

    encodings = tokenizer(inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    labels = tokenizer(targets, padding=True, truncation=True, max_length=max_length, return_tensors="pt").input_ids
    encodings["labels"] = labels
    return Dataset.from_dict(encodings)


# Step 3: Initialize Model and Tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# Step 4: Fine-tune the Model
def fine_tune_model(train_dataset):
    training_args = TrainingArguments(
        output_dir="./flan_t5_finetuned_2",
        per_device_train_batch_size=2,
        num_train_epochs=5,
        learning_rate=2e-4,
        warmup_steps=100,
        save_steps=10,
        save_total_limit=2,
        logging_dir="./logs",
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained("flan_t5_finetuned_2")


# Main function
def main():
    qa_pairs = get_qa_pairs()
    train_dataset = create_dataset(qa_pairs, tokenizer)
    fine_tune_model(train_dataset)

    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        inputs = tokenizer("question: " + question, return_tensors="pt", padding=True, truncation=True).input_ids
        outputs = model.generate(inputs)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
