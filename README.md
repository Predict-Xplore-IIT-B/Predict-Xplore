# Predict Xplore

Predict Xplore is a model-agnostic computer vision model deployment and testing platform designed to simplify the process of running inferences, applying explainable AI (XAI) techniques, and evaluating model performance. It provides a user-friendly interface and robust API for researchers and developers to deploy models, test datasets, and interpret results effectively.

## Demo Video

[![Demo](/thumbnail.png)](/predict%20Xplore%20nov25.mp4)

## Features

- **Explainable AI (XAI) Techniques**: Integrate cutting-edge XAI methods such as Grad-CAM, Ablation-CAM, and others to interpret and visualize model decisions, enhancing transparency and trust in AI systems.
- **Sequential Model Inference**: Run multiple models in sequence on the same dataset to compare performance, ensemble results, or apply cascading decision logic.
- **Role-Based Model Access**: Secure your models by assigning role-based access controls. Only users with the appropriate roles can access and utilize specific models, ensuring data privacy and compliance.
- **Comprehensive Inference Reports**: Receive detailed reports that include model predictions, labels, masks, and XAI visualizations. The reports are generated in a user-friendly PDF format and come with accompanying data files for deeper analysis.
- **Test Accuracy Calculation**: Upload datasets with test labels to automatically compute and receive test accuracy metrics per class, helping you assess model performance quantitatively.

## Getting Started

Once available, you can start using Predict Xplore by following these steps:

1. **Sign Up and Obtain an API Key**: Register on the Predict Xplore platform to receive your unique API key for authentication.
2. **Authenticate with the API**: Use the `/login` endpoint to obtain a session token by providing your API key.
3. **Upload Your Data**: Upload your images or datasets using the `/upload` endpoint. You can upload a single file or a zip archive containing multiple files and labels.
4. **Run Inference**: Use the `/test` endpoint to run inference on your data with the selected models and optional XAI methods.
5. **Retrieve Results**: Download the comprehensive report and accompanying data using the download link provided in the response.

## API Documentation

For detailed information on how to use the Predict Xplore API, please refer to the [API Documentation](API.md). The documentation includes:

- **Authentication**: How to authenticate and manage session tokens.
- **Endpoints**: Detailed descriptions of all API endpoints, including request parameters, headers, and example requests/responses.
- **Error Handling**: Information on possible error responses and how to handle them.
- **Usage Examples**: Sample code snippets and workflows to help you integrate the API into your applications.

## License

TODO: Add license.
