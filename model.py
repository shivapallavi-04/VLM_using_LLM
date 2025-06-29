# BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 24  # Maximum question length

# Build answer vocabulary
def build_vocab(texts, min_freq=1):
    from collections import Counter
    counter = Counter()
    for text in texts:
        tokens = tokenizer.tokenize(text)
        counter.update(tokens)
    vocab = {"<unk>": 0, "<pad>": 1, "<sos>": 2, "<eos>": 3}
    index = 4
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = index
            index += 1
    return vocab

vocab_answers = build_vocab(dataframe['response'])
answers_vocab_size = len(vocab_answers)
idx2word_answers = {idx: word for word, idx in vocab_answers.items()}

# Text to tensor for answers
def text_to_tensor(text, vocab, max_len):
    tokens = ["<sos>"] + tokenizer.tokenize(text)[:max_len-2] + ["<eos>"]
    indices = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    if len(indices) < max_len:
        indices += [vocab["<pad>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return torch.tensor(indices, dtype=torch.long)

# Custom VQA Dataset
class VQADataset(Dataset):
    def __init__(self, csv_path, image_folder, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df['image_id'] = self.df['image_id'] + '.png'
        self.image_folder = image_folder
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = 24

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_folder, row['image_id'])
        # Tokenize question for BERT
        question = self.tokenizer(row['question'], padding='max_length', max_length=self.max_len,
                                 truncation=True, return_tensors='pt')
        answer = text_to_tensor(row['response'], vocab_answers, 36)
        if os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        else:
            img = torch.zeros((3, 224, 224))
            print(f"Ảnh không tồn tại: {image_path}")
        return img, question['input_ids'].squeeze(0), question['attention_mask'].squeeze(0), answer

# Create DataLoaders
train_dataset = VQADataset(train_path, image_path, transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
eval_dataset = VQADataset(eval_path, image_path, transform)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=True, num_workers=4)

# CNN Feature Extractor
class CNN_Feature_Extractor_pretrained(nn.Module):
    def __init__(self):
        super(CNN_Feature_Extractor_pretrained, self).__init__()
        resnet = models.resnet50(weights=None)
        weights_path = "/kaggle/input/resnet/pytorch/default/1/resnet50-11ad3fa6.pth"
        state_dict = torch.load(weights_path)
        resnet.load_state_dict(state_dict)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, 512)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

# BERT-based Question Encoder
class Question_Encoder(nn.Module):
    def __init__(self):
        super(Question_Encoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, 512)  # Map BERT's output to 512-dim

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Use [CLS] token embedding
        return self.fc(pooled_output)  # Shape: [batch_size, 512]

# Attention 
class Attention(nn.Module):
    def __init__(self, hidden_dim=512):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, combined_feat):
        if hidden.dim() > 2:
            hidden = hidden.squeeze(0)
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        energy = torch.tanh(self.attn(torch.cat((hidden, combined_feat), dim=1)))
        attention_weights = F.softmax(self.v(energy), dim=1)
        context = attention_weights * combined_feat
        return context, attention_weights

# Answer Decoder
class Answer_Decoder(nn.Module):
    def __init__(self, answer_vocab_size, embedding_size=256, hidden_dim=512, k_beam=3):
        super(Answer_Decoder, self).__init__()
        self.embedding = nn.Embedding(answer_vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size + 1024, hidden_dim, num_layers=3, dropout=0.2, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, answer_vocab_size)
        self.k_beam = k_beam

    def forward(self, question_feat, image_feat, answer_seq=None, answer_maxlength=36):
        combined_feat = torch.cat((question_feat, image_feat), dim=1)
        if answer_seq is not None:
            x = self.embedding(answer_seq)
            hidden_state = None
            outputs = []
            for i in range(x.size(1)):
                context, _ = self.attention(hidden_state[0][-1] if hidden_state else question_feat, combined_feat)
                lstm_input = torch.cat((x[:, i, :], context), dim=1).unsqueeze(1)
                output, hidden_state = self.lstm(lstm_input, hidden_state)
                outputs.append(self.fc(output.squeeze(1)))
            return torch.stack(outputs, dim=1)
        else:
            batch_size = combined_feat.size(0)
            device = image_feat.device
            end_token = 3
            all_results = []
            for b in range(batch_size):
                b_question_feat = question_feat[b:b+1]
                b_combined_feat = combined_feat[b:b+1]
                beams = [(torch.tensor([[2]], dtype=torch.long, device=device), 0.0, None)]
                completed_beams = []
                for _ in range(answer_maxlength):
                    candidates = []
                    for seq, score, hidden_state in beams:
                        if seq[0, -1].item() == end_token:
                            completed_beams.append((seq, score, hidden_state))
                            continue
                        x = self.embedding(seq[:, -1])
                        prev_hidden = hidden_state[0][-1] if hidden_state else b_question_feat
                        context, _ = self.attention(prev_hidden, b_combined_feat)
                        lstm_input = torch.cat((x, context), dim=1).unsqueeze(1)
                        output, new_hidden = self.lstm(lstm_input, hidden_state)
                        logits = self.fc(output.squeeze(1))
                        log_probs = F.log_softmax(logits, dim=1)
                        topk_log_probs, topk_indices = log_probs.topk(self.k_beam)
                        for i in range(self.k_beam):
                            next_token = topk_indices[:, i:i+1]
                            next_score = score + topk_log_probs[:, i].item()
                            next_seq = torch.cat([seq, next_token], dim=1)
                            candidates.append((next_seq, next_score, new_hidden))
                    if not candidates:
                        break
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    beams = candidates[:self.k_beam]
                    if all(beam[0][0, -1].item() == end_token for beam in beams):
                        completed_beams.extend(beams)
                        break
                if completed_beams:
                    completed_beams.sort(key=lambda x: x[1], reverse=True)
                    best_seq = completed_beams[0][0]
                else:
                    beams.sort(key=lambda x: x[1], reverse=True)
                    best_seq = beams[0][0]
                all_results.append(best_seq)
            max_len = max(seq.size(1) for seq in all_results)
            padded_results = []
            for seq in all_results:
                if seq.size(1) < max_len:
                    padding = torch.full((1, max_len - seq.size(1)), end_token, dtype=torch.long, device=device)
                    padded_seq = torch.cat([seq, padding], dim=1)
                    padded_results.append(padded_seq)
                else:
                    padded_results.append(seq)
            return torch.cat(padded_results, dim=0)

# VQA Model with BERT
class VQA_Model(nn.Module):
    def __init__(self, answers_vocab_size, k_beam=3):
        super(VQA_Model, self).__init__()
        self.image_encoder = CNN_Feature_Extractor_pretrained().to(device)
        self.question_encoder = Question_Encoder().to(device)
        self.answer_decoder = Answer_Decoder(answers_vocab_size, k_beam=k_beam).to(device)

    def forward(self, image, input_ids, attention_mask, answer_seq=None):
        image_feat = self.image_encoder(image)
        question_feat = self.question_encoder(input_ids, attention_mask)
        output = self.answer_decoder(question_feat, image_feat, answer_seq)
        return output