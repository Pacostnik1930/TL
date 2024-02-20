from transformers import BertTokenizer, BertForSequenceClassification

   # Загрузка предобученной модели BERT для классификации
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

   # Процедура классификации текста
text = "Шла Саша по шоссе и сосала сушку"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)