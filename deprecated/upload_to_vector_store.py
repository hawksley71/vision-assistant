import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

def upload_detection_logs(file_path="detection_logs_for_rag.csv", output_txt="outputs/vector_store_info.txt"):
    """
    Uploads the detection logs file to OpenAI for use with an assistant.
    Handles both .csv (for code interpreter) and .txt (for file_search/vector store).
    Writes the file ID(s) and vector store ID (if applicable) to a text file in the outputs folder.
    """
    _ = load_dotenv(find_dotenv())
    client = OpenAI(
        api_key=os.getenv("OPENAI_TOKEN"),
        default_headers={"OpenAI-Beta": "assistants=v2"}
    )

    file_ext = os.path.splitext(file_path)[1].lower()
    file_ids = []
    vector_store_id = None

    if file_ext == ".csv":
        print(f"Uploading {file_path} as a CSV for code interpreter...")
        with open(file_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="assistants")
            file_ids.append(file_obj.id)
        print(f"File uploaded. File ID: {file_obj.id}")
    elif file_ext == ".txt":
        print(f"Uploading {file_path} as a TXT for file_search/vector store...")
        with open(file_path, "rb") as f:
            vector_store = client.vector_stores.create(name="Detection Logs Vector Store")
            file_batch = client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id,
                files=[f]
            )
        vector_store_id = vector_store.id
        files = client.vector_stores.files.list(vector_store_id=vector_store_id)
        file_ids = [file.id for file in files.data]
        print(f"Files uploaded to vector store: {file_batch.status}")
        print(f"File IDs: {file_ids}")
    else:
        raise ValueError("Unsupported file type. Only .csv and .txt are supported.")

    # Write info to output file
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w") as out_f:
        out_f.write(f"file_ids: {file_ids}\n")
        if vector_store_id:
            out_f.write(f"vector_store_id: {vector_store_id}\n")
    print(f"File and vector store IDs written to {output_txt}")
    return file_ids, vector_store_id

if __name__ == "__main__":
    file_ids, vector_store_id = upload_detection_logs("outputs/combined_logs.csv") 