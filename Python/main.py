from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from ml.preprocess import preprocess_data, save_preprocessed_data, suggest_missing_strategy
from ml.models import train_model, save_model
from ml.utils import get_dataset_insights
from dotenv import load_dotenv
import uvicorn

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_file(files: list[UploadFile] = File(...)):
    try:
        results = {}
        os.makedirs("uploads", exist_ok=True)
        for file in files:
            file_location = f"uploads/{file.filename}"
            with open(file_location, "wb") as f:
                f.write(await file.read())
            
            df = pd.read_csv(file_location)
            summary = {
                "columns": list(df.columns),
                "rows": len(df),
                "data_types": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "unique_values": {col: df[col].nunique() for col in df.columns},  # Added unique value counts
                "stats": {
                    col: df[col].describe().to_dict() 
                    for col in df.select_dtypes(include=['float64', 'int64']).columns  # Descriptive stats for numeric columns
                }
            }
            insights_data = get_dataset_insights(summary, df)
            suggested_missing_strategy = suggest_missing_strategy(df)
            print(f"File: {file.filename}, Suggested missing strategy: {suggested_missing_strategy}")
            print(f"File: {file.filename}, Suggested task type: {insights_data['suggested_task_type']}")
            print(f"File: {file.filename}, Suggested target column: {insights_data['suggested_target_column']}")
            results[file.filename] = {
                "summary": summary,
                "insights": insights_data["insights"],
                "suggested_task_type": insights_data["suggested_task_type"],
                "suggested_target_column": insights_data["suggested_target_column"],
                "suggested_missing_strategy": suggested_missing_strategy
            }
        return JSONResponse(content=results)
    except Exception as e:
        print(f"Error in /upload endpoint: {str(e)}")
        return JSONResponse(content={"error": f"Upload failed: {str(e)}"}, status_code=500)

@app.post("/preprocess")
async def preprocess_endpoint(
    files: list[UploadFile] = File(...),
    missing_strategy: str = Form(...),
    scaling: bool = Form(...),
    encoding: str = Form(...),
    target_column: str = Form(None)  
):
    try:
        results = {}
        os.makedirs("uploads", exist_ok=True)
        for file in files:
            file_location = f"uploads/{file.filename}"
            with open(file_location, "wb") as f:
                f.write(await file.read())
            
            df = pd.read_csv(file_location)
            # Validate encoding and target_column compatibility
            if encoding in ["target", "kfold"] and (not target_column or target_column not in df.columns):
                raise ValueError(f"Target column '{target_column}' is required and must exist in the dataset for {encoding} encoding")
            
            df_processed = preprocess_data(df, missing_strategy=missing_strategy, scaling=scaling, encoding=encoding, target_column=target_column)
            preprocessed_file = save_preprocessed_data(df_processed, filename=f"preprocessed_{file.filename}")
            results[file.filename] = {"preprocessed_file": preprocessed_file}
        return JSONResponse(content=results)
    except Exception as e:
        print(f"Error in /preprocess endpoint: {str(e)}")
        return JSONResponse(content={"error": f"Preprocessing failed: {str(e)}"}, status_code=500)


@app.post("/train")
async def train_model_endpoint(
    preprocessed_filenames: list[str] = Form(...),
    target_column: str = Form(None),
    task_type: str = Form(...),
    model_type: str = Form(None)
):
    try:
        results = {}
        for filename in preprocessed_filenames:
            file_location =     filename 
            if not os.path.exists(file_location):
                return JSONResponse(content={"error": f"Preprocessed file {filename} not found"}, status_code=404)
            df_processed = pd.read_csv(file_location)
            result = train_model(df_processed, target_column, task_type, model_type)
            if "model" in result:
              
                basename = os.path.basename(filename)  
                original_filename = basename.replace("preprocessed_", "", 1)  
                model_filename = f"trained_model_{original_filename.split('.')[0]}.pkl"  
                save_model(result["model"], file_path=f"uploads/{model_filename}")
                del result["model"]
            results[filename] = result
        return JSONResponse(content=results)
    except Exception as e:
        print(f"Error in /train endpoint: {str(e)}")
        return JSONResponse(content={"error": f"Training failed: {str(e)}"}, status_code=500)
    

    
@app.get("/download-model/{filename}")
async def download_model(filename: str):
    try:
        model_file = f"uploads/trained_model_{filename.split('.')[0]}.pkl"
        if os.path.exists(model_file):
            return FileResponse(model_file, filename=f"trained_model_{filename.split('.')[0]}.pkl")
        return JSONResponse(content={"error": "Model file not found"}, status_code=404)
    except Exception as e:
        print(f"Error in /download-model endpoint: {str(e)}")
        return JSONResponse(content={"error": f"Download failed: {str(e)}"}, status_code=500)

@app.get("/download-preprocessed/{filename}")
async def download_preprocessed(filename: str):
    try:
        preprocessed_file = f"uploads/preprocessed_{filename}"
        if os.path.exists(preprocessed_file):
            return FileResponse(preprocessed_file, filename=f"preprocessed_{filename}")
        return JSONResponse(content={"error": "Preprocessed file not found"}, status_code=404)
    except Exception as e:
        print(f"Error in /download-preprocessed endpoint: {str(e)}")
        return JSONResponse(content={"error": f"Download failed: {str(e)}"}, status_code=500)
    

if __name__ == "__main__":
    uvicorn.run(app, port=8000)