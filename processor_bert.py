from sentence_transformers import SentenceTransformer
import joblib

transformer_model=SentenceTransformer('all-MiniLM-L6-v2')
classifier_model=joblib.load('models/log_classifier.joblib')


def classify_with_bert(log_message):
    # Generate BERT embeddings for the log message
    message_embeddings = transformer_model.encode(log_message)
    probabilities=classifier_model.predict_proba([message_embeddings])[0]
    predicted_class="Unknown"
    if max(probabilities)>0.5:
        predicted_class = classifier_model.predict([message_embeddings])[0]
    return predicted_class  # Return the first prediction

    
if __name__=="__main__":
    logs=[
        "User User123 logged in.",
        "Backup started at 2023-10-01 12:00:00.",
        "nova.osapi_compute.wsgi.server [req-4d05bae9-8a34-42b6-98be-991eae0f992c 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] 10.11.10.1 'GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1' Return code: 200 len: 1874 time: 0.1925120",
        "System updated to version 1.2.3.",
        "File report.pdf uploaded successfully by user User456.",
        "Disk cleanup completed successfully.",
        "System reboot initiated by user User789.",
        "Account with ID 12345 created by admin."
    ]

    for log in logs:
        label=classify_with_bert(log)
        print(f"Log: {log}\nPredicted Label: {label}\n")