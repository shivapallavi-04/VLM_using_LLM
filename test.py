# Test model
def test_model(model, question, image_path, ground_truth, idx2word):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    question_inputs = tokenizer(question, padding='max_length', max_length=24, truncation=True, return_tensors='pt')
    input_ids = question_inputs['input_ids'].to(device)
    attention_mask = question_inputs['attention_mask'].to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor, input_ids, attention_mask)
    predicted_answer = tensor_to_text(output, idx2word)[0]
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Q: {question}\nPredicted answer: {predicted_answer}\nGround Truth: {ground_truth}", fontsize=12)
    plt.show()
    return predicted_answer

def test_random_samples(model, eval_dataframe, idx2word):
    samples = eval_dataframe.sample(n=15)
    for index, row in samples.iterrows():
        question = row['question']
        image_path = f'/kaggle/input/visual-question-answering-computer-vision-nlp/dataset/images/{row["image_id"]}.png'
