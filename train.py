# Training function
def train_model(model, train_loader, eval_loader, criterion, optimizer, best_model_path, num_epochs=10, patience=5):
    import time
    model.to(device)
    best_loss = float('inf')
    no_improve_epochs = 0
    history = {"train_loss": [], "eval_loss": [], "bleu_score": []}
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        for images, input_ids, attention_mask, answers in train_loader:
            images, input_ids, attention_mask, answers = images.to(device), input_ids.to(device), attention_mask.to(device), answers.to(device)
            optimizer.zero_grad()
            output = model(images, input_ids, attention_mask, answers[:, :-1])
            loss = criterion(output.view(-1, output.size(-1)), answers[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        model.eval()
        eval_loss = 0
        bleu_scores = []
        with torch.no_grad():
            for images, input_ids, attention_mask, answers in eval_loader:
                images, input_ids, attention_mask, answers = images.to(device), input_ids.to(device), attention_mask.to(device), answers.to(device)
                output = model(images, input_ids, attention_mask, answers[:, :-1])
                loss = criterion(output.view(-1, output.size(-1)), answers[:, 1:].reshape(-1))
                eval_loss += loss.item()
                predicted_answers = tensor_to_text(model(images, input_ids, attention_mask), idx2word_answers)
                answers_text = tensor_to_text(answers, idx2word_answers)
                bleu = compute_bleu(predicted_answers, answers_text)
                bleu_scores.append(bleu)
        avg_eval_loss = eval_loss / len(eval_loader)
        avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
        end_time = time.time()
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Evaluation Loss: {avg_eval_loss:.4f}, BLEU Score: {avg_bleu_score:.4f}")
        history["train_loss"].append(avg_train_loss)
        history["eval_loss"].append(avg_eval_loss)
        history["bleu_score"].append(avg_bleu_score)
        if avg_eval_loss < best_loss:
            best_loss = avg_eval_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_model_path)
            print("Best model saved!")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs}/{patience} epochs.")
            if no_improve_epochs >= patience:
                print("Early stopping triggered!")
                break
    return history

# BLEU score functions
def ngram_precision(reference, candidate, n):
    from collections import Counter
    ref_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference)-n+1)])
    cand_ngrams = Counter([tuple(candidate[i:i+n]) for i in range(len(candidate)-n+1)])
    overlap = sum(min(cand_ngrams[ngram], ref_ngrams.get(ngram, 0)) for ngram in cand_ngrams)
    total = sum(cand_ngrams.values())
    return overlap / total if total > 0 else 0

def brevity_penalty(reference, candidate):
    ref_len = len(reference)
    cand_len = len(candidate)
    if cand_len > ref_len:
        return 1
    else:
        return math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0

def compute_bleu(reference_sentences, candidate_sentences, max_n=4):
    assert len(reference_sentences) == len(candidate_sentences)
    bleu_scores = []
    for ref, cand in zip(reference_sentences, candidate_sentences):
        precisions = [ngram_precision(ref, cand, n) for n in range(1, max_n+1)]
        geometric_mean = math.exp(sum(math.log(p) for p in precisions if p > 0) / max_n) if any(precisions) else 0
        bp = brevity_penalty(ref, cand)
        bleu_scores.append(bp * geometric_mean)
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

def tensor_to_text(tensor, idx2word):
    sentences = []
    for seq in tensor:
        words = [idx2word[idx.item()] for idx in seq if idx.item() in idx2word]
        if "<sos>" in words:
            words.remove("<sos>")
        sentence = " ".join(words).split("<eos>")[0]
        sentences.append(sentence.strip())
    return sentences

# Initialize and train model
VQA_model = VQA_Model(answers_vocab_size)
criterion = nn.CrossEntropyLoss(ignore_index=1)
optimizer = AdamW(VQA_model.parameters(), lr=1e-4, weight_decay=1e-2)
VQA_model_history = train_model(VQA_model, train_loader, eval_loader, criterion, optimizer, '/kaggle/working/VAQ_model_bert.pth', num_epochs=50)
