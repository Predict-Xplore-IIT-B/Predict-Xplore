# Predict Xplore API

This document describes the Predict Xplore API.

- [Base URL](#base-url)
- [Authentication](#authentication)
  - [Session Token Management](#session-token-management)
- [Endpoints](#endpoints)
  - [POST `/login`](#1-post-login)
  - [POST `/test`](#2-post-test)
  - [POST `/upload`](#3-post-upload)
  - [GET `/models`](#4-get-models)
  - [GET `/model`](#5-get-model)

## Base URL

The planned base URL is **https://www.predictxplore.com/api/v1**. Please note that this URL is subject to change.

## Authentication

To access this API, users must authenticate using an API key. The API requires an initial login to generate a session token, which is then used to authorize all subsequent requests.

### Session Token Management

- The session token is required for all API requests except for `/login`.
- Tokens expire after a set period. Users need to log in again to obtain a new token when their session expires.
- Store tokens securely and avoid exposing them in client-side code or browser storage.

For all API endpoints requiring authentication, besides `/login`, if the session token is invalid, expired, or missing, the following error code and response format are returned:

- **401 Unauthorized:**
  ```json
  {
    "error": "Invalid or expired session token. Please log in again to obtain a new session token."
  }
  ```

### 1.  **Upload Image:** First, you send a `POST` request to `/model/instance/upload` with your image. The server saves the image and returns a unique `test_case_id`.
### 2.  **Run Prediction:** Next, you send a `POST` request to `/model/instance/predict`, including the `test_case_id` you just received. The server runs the model(s), generates the report, and returns a `download_url`.

Ensure that session tokens are stored securely and refreshed as needed by re-authenticating when a 401 error occurs.

## Endpoints

### 1. POST `/login`

This endpoint generates a session token for authorized users. The session token should be included in the `Authorization` header of all subsequent API calls.

#### Headers

- **`Content-Type`**: `application/json`

#### Request Parameters

- **`api_key`** (*required*, *string*): The user’s API key for authentication.

#### Response

- **200 OK**:
  - `session_token`: A session token that should be included in the `Authorization` header for subsequent API calls.
  - `expires_after`: Duration in seconds until the session token expires.

#### Error Responses

- **401 Unauthorized**: Invalid API key.

#### Usage Example

```http
POST /login
Content-Type: application/json

{
  "api_key": "<YOUR_API_KEY>"
}
```

**Response:**

```json
{
  "session_token": "your_generated_session_token",
  "expires_after": 3600
}
```

**Note**: The session token will need to be refreshed by re-authenticating with the API key once the `expires_after` (3600 seconds or 1 hour in the above example) duration has passed.

---

### 2. POST `/test`

This endpoint runs inference on input data using specified computer vision models and optionally applies explainable AI (XAI) techniques to selected target classes. Users receive a link to download a detailed report upon successful completion.

#### Headers

- **`Authorization`** (*required*, *string*): Bearer token in the format `Bearer <SESSION_TOKEN>`.
- **`Content-Type`**: `application/json`

#### Request Parameters

- **`model_ids`** (*required*, *list of strings*): A list of computer vision model IDs to apply. You can specify one or more model IDs.
- **Input Source** (*one of the following required*):
  - **`directory_path`** (*string*): Absolute path to a directory on the server containing images and optionally a JSON file with labels/masks in COCO format. Files can be uploaded to the server using the [`/upload`](#3-post-upload) method.
  - **`image_path`** (*string*): Absolute path to a single image file on the server or URL to an image. Files can be uploaded to the server using the [`/upload`](#3-post-upload) method.
  - **`video_link`** (*string*): Link to a video feed; frames will be extracted for analysis.
- **`xai_methods`** (*optional*, *list of strings*): List of XAI techniques to apply. If omitted, XAI methods are not run.
- **`target_classes`** (*optional*, *list of strings*): List of target classes for applying XAI techniques. If omitted, XAI runs on all detected classes.

#### Output

Upon success, the API returns a downloadable directory containing the following files:

- **PDF Report**: Detailed report of model results.
- **COCO JSON File**: JSON file with labels, masks, or segments.
- **Test Accuracy (Optional)**: If input data includes labeled data, a `results.txt` file is generated with computed accuracy per class in the format `test_accuracy=value`.
- **XAI Results Directory (Optional)**: Directory with XAI results, including heatmap images or coordinate files, depending on the XAI techniques applied.

The directory structure is as follows:

```txt
- report.pdf
- labels.json
- results.txt [*]
- xai/ [*]
  ├── image_name_xai_method_target_class.png [*]
  └── image_name_xai_method_target_class.txt [*]
```

Files and directories marked with `[*]` are optional.

#### Response

- **200 OK**:
  - `download_link`: URL to download the zipped directory containing all outputs.
  - `test_accuracy` (if computed): Test accuracy of the models on the provided data.

- **Error Responses**:
  - **400 Bad Request**: Model IDs or XAI methods are invalid, or target classes are incompatible with the models.
  - **[401 Unauthorized](#session-token-management)**.
  - **422 Unprocessable Entity**: Missing or invalid request parameters.

#### Example Request

```http
POST /test
Authorization: Bearer <SESSION_TOKEN>
Content-Type: application/json

{
  "model_ids": ["model_123"],
  "directory_path": "/data/input_dir",
  "xai_methods": ["gradcam"],
  "target_classes": ["path", "water-body"]
}
```

---

### 3. POST `/upload`

This endpoint allows users to upload files to the server. Users can upload a single file (e.g., an image) or a zip file containing multiple files, such as images and label files. The server stores the uploaded file(s) in a secure, sandboxed environment for processing by other API methods like [`/test`](#2-post-test).

#### Headers

- **`Authorization`** (*required*, *string*): Bearer token in the format `Bearer <SESSION_TOKEN>`.
- **`Content-Type`**: `multipart/form-data`

#### Request Parameters

You must include one of the following in your multipart form data:

- **`file`** (*required if `zip_file` not provided*, *file*): A single file to upload. This can be an image file or any other supported file type.
- **`zip_file`** (*required if `file` not provided*, *file*): A zip file containing images, labels, masks, or segments in COCO format.

#### Response

- **200 OK**:

  - If **`file`** was uploaded:
    - `file_path`: Path to the uploaded file on the server. This path can be used directly in the `image_path` parameter of the [`/test`](#2-post-test) endpoint.
    - `deleted_after` (*nullable*): Duration in seconds after which the uploaded file will be automatically deleted from the server. It can be `null`, meaning the files will not be deleted.
    - *Example:*
      ```json
      {
        "file_path": "/uploads/user_1234/session_5678/image.jpg",
        "deleted_after": 86400
      }
      ```
  - If **`zip_file`** was uploaded:
    - `directory_path`: Path to the directory on the server where your uploaded files are stored. This path can be used directly in the `directory_path` parameter of the [`/test`](#2-post-test) endpoint.
    - `deleted_after` (*nullable*): Duration in seconds after which the uploaded file will be automatically deleted from the server. It can be `null`, meaning the files will not be deleted.
    - *Example:*
      ```json
      {
        "directory_path": "/uploads/user_1234/session_5678/dirname/",
        "deleted_after": null
      }
      ```

- **Error Responses**:
  - **400 Bad Request**: No file provided, invalid file type, or both `file` and `zip_file` are provided.
    - *Example:*
      ```json
      {
        "error": "No file provided or unsupported file type. Please upload a valid file."
      }
      ```
  - **[401 Unauthorized](#session-token-management)**.
  - **413 Payload Too Large**: Uploaded file exceeds the server's maximum file size limit.
    - *Example:*

      ```json
      {
        "error": "Uploaded file exceeds the maximum allowed size of 100MB."
      }
      ```
  - **422 Unprocessable Entity**: File upload failed, or content structure is invalid (e.g., missing required files in a zip archive).
    - *Example:*
      ```json
      {
        "error": "File upload failed or the uploaded zip file is missing required contents."
      }
      ```

#### Usage Notes

- **Single File Upload (`file`)**:
  - Use this option when you need to upload a single image or data file.
  - The uploaded file will be stored in a dedicated directory on the server.
  - The API response will include `file_path`, which points directly to your uploaded file. Use this `file_path` in the `image_path` parameter of the [`/test`](#2-post-test) endpoint.

- **Multiple Files Upload (`zip_file`)**:
  - Use this option to upload multiple files at once by compressing them into a zip archive.
  - The API response will include `directory_path`, which you can use in the `directory_path` parameter of the [`/test`](#2-post-test) endpoint.

- **Path Security**:
  - Treat the `file_path` and `directory_path` as sensitive information.
  - Do not expose them in client-side code, logs, or error messages.
  - Store them securely in your application's backend if needed.

- In JSON, `null` is different from the string `"null"`.

#### Example Requests

**Uploading a Single File:**

```http
POST /upload
Authorization: Bearer <SESSION_TOKEN>
Content-Type: multipart/form-data

(Form Data)
- file: <Your Image File>
```

**Uploading a Zip File:**

```http
POST /upload
Authorization: Bearer <SESSION_TOKEN>
Content-Type: multipart/form-data

(Form Data)
- zip_file: <Your ZIP File>
```

#### Example Responses

**Successful Single File Upload:**

```json
{
  "file_path": "/uploads/user_1234/session_5678/image.jpg",
  "deleted_after": 86400
}
```

**Successful Zip File Upload:**

```json
{
  "directory_path": "/uploads/user_1234/session_5678/dirname/",
  "deleted_after": 86400
}
```

#### Important Notes

- **Mutual Exclusivity**:
  - You must provide either `file` or `zip_file` in your request, but not both.
  - If both are provided, the server will return a **400 Bad Request** error.
- **Supported File Types**:
  - For `file`, supported types include common image formats (e.g., JPEG, PNG) and other data files compatible with the `/test` endpoint.
  - For `zip_file`, ensure the archive is in `.zip` format and that it contains supported file types.
- **Content Structure**:
  - When uploading a zip file, maintain a consistent directory structure if required by the `/test` endpoint.
  - Include any necessary annotation files (e.g., labels in COCO format) within the zip archive.
- **File Size Limits**:
  - Be mindful of the server's maximum file size limits to avoid **413 Payload Too Large** errors.
- **Data Retention**:
  - Uploaded files and directories are temporary and will be deleted after the `deleted_after` duration elapses.
  - If you need to retain data longer, you will need to re-upload the files when needed.

---

### 4. GET `/models`

Returns a list of all available model IDs.

#### Headers

- **`Authorization`** (*required*, *string*): Bearer token in the format `Bearer <SESSION_TOKEN>`.

#### Response

- **200 OK**:
  - `model_ids`: Array of all model IDs currently available for use.
  - Example Response:
    ```json
    {
      "model_ids": ["model1", "model3"]
    }
    ```
    The array can be empty if no models are available.
- **Error Responses**:
  - **[401 Unauthorized](#session-token-management)**.

---

### 5. GET `/model`

Returns detailed information about a specified model based on optional parameters. 

#### Headers

- **`Authorization`** (*required*, *string*): Bearer token in the format `Bearer <SESSION_TOKEN>`.

#### Query Parameters

- **`id`** (*required*, *string*): Model ID.
  
- **Optional Parameters**:
  - **`xai`** (*optional*, *boolean*, *default: false*): If true, includes a list of compatible XAI methods for the model.
  - **`description`** (*optional*, *boolean*, *default: false*): If true, includes a description of the model.
  - **`target_classes`** (*optional*, *boolean*, *default: false*): If true, lists target classes available for the model.
  - **`type`** (*optional*, *boolean*, *default: false*): If true, specifies the model type (e.g., detection, segmentation).

**Note:** The optional parameter values are case-insensitive and do *not* accept other truthy values like `1` or `yes`.

#### Response

- **200 OK**: Returns details for the specified model based on the optional parameters provided.

- **Error Responses**:
  - **400 Bad Request**: Missing model ID or invalid parameters.
    - Example:
      ```json
      {
        "error": "Model ID is required or optional parameters are invalid."
      }
      ```
  - **[401 Unauthorized](#session-token-management)**.
  - **404 Not Found**: Model ID does not exist.
    - Example:
      ```json
      {
        "error": "Model not found for the provided ID."
      }
      ```

#### Example Request

```http
GET /model?id=model_123&description=true&type=true&xai=true
Authorization: Bearer <SESSION_TOKEN>
```

**Response:**
```json
{
  "model_id": "model_123",
  "description": "Model for forest image segmentation",
  "type": "segmentation",
  "xai_methods": ["gradcam", "ablationcam"]
}
```

#### Full Response Schema

```json
{
  "model_id": "string",
  "description": "optional, string",
  "type": "optional, string",
  "xai_methods": ["optional", "list", "of", "strings"],
  "target_classes": ["optional", "list", "of", "strings"]
}
```
POST /model/instance/upload
Initiates an inference test by uploading an image. The server creates a TestCase record in the database and returns its unique ID, which is required for the next step.

Headers
Authorization (required, string): The user's authentication token.

Content-Type: multipart/form-data

Body (Multipart Form)
image (required, file): The image file to be analyzed.

Success Response (201 Created)
test_case_id: A unique integer identifying this test run.

Example Response
{
    "message": "Image uploaded successfully and saved to a test case.",
    "test_case_id": 1
}

POST /model/instance/predict
Runs the selected model(s) and optional XAI algorithms on the image associated with the provided test_case_id.

Headers
Authorization (required, string): The user's authentication token.

Content-Type: application/json

Request Body (JSON)
test_case_id (required, integer): The ID returned from the /model/instance/upload endpoint.

models (required, list of strings): A list of model names to run (e.g., ["HumanDetection"]).

xai_algo (optional, string): The XAI algorithm to apply (e.g., "gradcam").

target_class (optional, string): The specific class to explain for segmentation models (e.g., "forest").

Success Response (200 OK)
reports: A list of objects, each containing a link to download a generated PDF report.

Example Response
{
    "message": "Inference and report generation complete.",
    "reports": [
        {
            "model_name": "HumanDetection",
            "download_url": "http://127.0.0.1:8000/model/download/report/1"
        }
    ]
}

GET /model/download/report/{report_id}
Downloads the final PDF report generated by the /predict endpoint.

URL Parameters
report_id (required, integer): The unique ID of the report, from the download_url in the previous step.

Headers
Authorization (required, string): The user's authentication token.

Success Response (200 OK)
The server responds with the PDF file, which your API client will prompt you to save.