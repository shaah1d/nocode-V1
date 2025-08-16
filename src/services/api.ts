import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000";

// Utility function to handle errors
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const handleError = (error: any) => {
  if (error.response) {
    // The request was made and the server responded with a status code
    // that falls out of the range of 2xx
    console.error("Response error:", error.response.data);
    throw new Error(
      error.response.data.error || "Server responded with an error"
    );
  } else if (error.request) {
    // The request was made but no response was received
    console.error("Request error:", error.request);
    throw new Error("No response received from the server");
  } else {
    // Something happened in setting up the request that triggered an Error
    console.error("Error:", error.message);
    throw new Error(error.message);
  }
};

// Optimized upload files function
const uploadFiles = async (
  files: File[],
  onProgress: (progress: number) => void
) => {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  onProgress(30);
  console.log("Sending upload request to http://127.0.0.1:8000/upload...");
  try {
    const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    console.log(response.data);
    onProgress(100);
    return response.data;
  } catch (error) {
    handleError(error);
    return {}; // Add this to prevent TypeScript error
  }
};

// Updated preprocess function to include target_column
const preprocessData = async (
  files: File[],
  missingStrategy: string,
  scaling: boolean,
  encoding: string,
  targetColumn = "",
  onProgress: (progress: number) => void
) => {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  formData.append("missing_strategy", missingStrategy);
  formData.append("scaling", scaling.toString());
  formData.append("encoding", encoding);

  // Add target column if provided (needed for target encoding)
  if (targetColumn) {
    formData.append("target_column", targetColumn);
  }

  onProgress(40);
  console.log(
    "Sending preprocess request to http://127.0.0.1:8000/preprocess..."
  );
  try {
    const response = await axios.post(`${API_BASE_URL}/preprocess`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 60000,
    });
    onProgress(100);
    return response.data;
  } catch (error) {
    handleError(error);
    return {}; 
  }
};

const trainModel = async (
  files: File[],
  targetColumn: string,
  taskType: string,
  modelType: string,
  onProgress: (progress: number) => void
) => {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  formData.append("target_column", targetColumn || "");
  formData.append("task_type", taskType);
  formData.append("model_type", modelType);

  onProgress(30);
  console.log("Sending train request to http://127.0.0.1:8000/train...");
  try {
    const response = await axios.post(`${API_BASE_URL}/train`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 60000,
    });
    onProgress(100);
    return response.data;
  } catch (error) {
    handleError(error);
    return {}; // Add this to prevent TypeScript error
  }
};

// Download URLs
const getDownloadModelUrl = (filename: string) => {
  return `${API_BASE_URL}/download-model/${filename}`;
};

const getDownloadPreprocessedUrl = (filename: string) => {
  return `${API_BASE_URL}/download-preprocessed/${filename}`;
};

const trainModelWithPreprocessed = async (
  preprocessedFilenames: string[],
  targetColumn: string,
  taskType: string,
  modelType: string,
  onProgress: (progress: number) => void
) => {
  const formData = new FormData();
  preprocessedFilenames.forEach((filename) =>
    formData.append("preprocessed_filenames", filename)
  );
  formData.append("target_column", targetColumn || "");
  formData.append("task_type", taskType);
  formData.append("model_type", modelType);

  onProgress(30);
  console.log("Sending train request to http://127.0.0.1:8000/train...");
  try {
    const response = await axios.post(`${API_BASE_URL}/train`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 60000,
    });
    onProgress(100);
    return response.data;
  } catch (error) {
    handleError(error);
    return {};
  }
};

export {
  uploadFiles,
  preprocessData,
  trainModel,
  getDownloadModelUrl,
  getDownloadPreprocessedUrl,
  trainModelWithPreprocessed,
};
