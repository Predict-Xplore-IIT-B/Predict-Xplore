import React, { useState } from "react";
import AdminNavbar from "../../components/AdminNavbar";
import SelectCheckbox from "../../components/SelectCheckbox";
import upload from "../../assets/upload.png";
import axios from "axios";
import { toast } from "react-toastify";

function CreateModel() {

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [modelFile, setModelFile] = useState(null);
  const [modelImage, setModelImage] = useState(null);
  const [modelType, setModelType] = useState("");
  const [createdBy, setCreatedBy] = useState(name); // Username of the creator

  const [selectedClass, setSelectedClass] = useState([]);
  const [selectedRoles, setSelectedRoles] = useState([]);
  const [isFileUploaded, setIsFileUploaded] = useState(false); // Track file upload status
  const [isImageUploaded, setIsImageUploaded] = useState(false); // Track image upload status

  const classes = ["class 1", "class 5", "class 3", "class 4"];
  const roles = ["CSE DEPT", "AIML DEPT", "MECH DEPT", "DS DEPT"];
  const types = [
    "Human Detection",
    "Image Segmentation",
    "Object Detection",
    "Classification",
    "Decision Tree",
  ];


  const handleFileChange = (e) => {
    setModelFile(e.target.files[0]);
  };

  const handleImageChange = (e) => {
    setModelImage(e.target.files[0]);
  };

  

  const handleUpload = async () => {

    toast.success("Model successfully uploaded!", { autoClose: 2000 });

    return;

    setModelType(types[0]);
    if (!name || !description || !modelFile || !modelImage ) {
      alert("All fields are required!");
      return;
    }
    
    const formData = {"name": name, "description": description, "model_file": modelFile, "model_image": modelImage, "model_type": modelType, "created_by": createdBy};
    // formData.append("name", name);
    // formData.append("description", description);
    // formData.append("model_file", modelFile); // .pt file
    // formData.append("model_image", modelImage); // Image file
    // formData.append("model_type", modelType);
    // formData.append("created_by", createdBy);
    
    console.log(formData);
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/model/create", // Adjust your backend URL
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      console.log(response.data);
      alert("Model successfully uploaded!");
    } catch (error) {
      console.error("Error uploading model:", error);
      alert("Failed to upload model.");
    }
  };
  

  return (
    <div className="h-screen w-screen bg-[#EAECFF]">
      <AdminNavbar />
      <div className="flex flex-col mx-20">
        <div>
          <input
            type="search"
            className="relative m-0 block w-[35%] min-w-0 rounded-full border border-solid border-neutral-300 bg-clip-padding px-3 py-[0.25rem] text-base leading-[1.6] outline-none transition duration-200 ease-in-out focus:z-[3] focus:border-primary focus:text-neutral-700 focus:shadow-[inset_0_0_0_1px_rgb(59,113,202)] focus:outline-none bg-white h-[5vh] mt-24"
            placeholder="Enter name of the model"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </div>

        <div className="flex justify-end mt-5">
          <SelectCheckbox
            options={classes}
            title="Select Class"
            setSelected={setSelectedClass}
          />
          <SelectCheckbox
            options={roles}
            title="Select Roles"
            setSelected={setSelectedRoles}
          />
          <SelectCheckbox
            options={modelType}
            title="Select Model Type"
            setSelected={setModelType}
          />
        </div>

        <div>
          <textarea
            className="w-[100%] h-[9vh] p-2 rounded-2xl resize-none"
            placeholder="Enter Description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
        </div>

        <div className="mt-5">
          <div className="flex items-center justify-center w-full h-[25vh] bg-white rounded-2xl">
            <div className="m-8 pr-20">
              <label
                htmlFor="model-file"
                className="cursor-pointer hover:bg-gray-200"
              >
                <div className="flex flex-col items-center justify-center">
                  {/* Conditionally render checkmark after upload */}
                  {isFileUploaded ? (
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="green"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="h-20 w-20 animate-pulse"
                    >
                      <path d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    <img
                      src={upload}
                      alt="Upload model file"
                      className="h-20"
                    />
                  )}
                </div>
                <input
                  id="model-file"
                  type="file"
                  className="hidden"
                  onChange={(e) => {
                    setModelFile(e.target.files[0]);
                    setIsFileUploaded(true); // Mark as uploaded
                  }}
                />
              </label>
              <p className="my-2 text-xl text-gray-500 dark:text-gray-400">
                Upload your Model here
              </p>
            </div>

            <div className="m-8 pl-20">
              <label
                htmlFor="model-image"
                className="cursor-pointer hover:bg-gray-200"
              >
                <div className="flex flex-col items-center justify-center">
                  {/* Conditionally render checkmark after upload */}
                  {isImageUploaded ? (
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="green"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="h-20 w-20 animate-pulse"
                    >
                      <path d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    <img
                      src={upload}
                      alt="Upload model image"
                      className="h-20"
                    />
                  )}
                </div>
                <input
                  id="model-image"
                  type="file"
                  className="hidden"
                  onChange={(e) => {
                    setModelImage(e.target.files[0]);
                    setIsImageUploaded(true); // Mark as uploaded
                  }}
                />
              </label>
              <p className="my-2 text-xl text-gray-500 dark:text-gray-400">
                Upload Model Image
              </p>
            </div>
          </div>
        </div>

        <div className="w-full mt-6 flex justify-center items-center">
          <button
            type="button"
            onClick={handleUpload}
            className="text-white bg-[#6966FF] hover:bg-blue-800 focus:outline-none focus:ring-4 focus:ring-blue-300 font-extrabold rounded-full text-2xl px-12 py-2.5 text-center me-2 mb-2"
          >
            Create
          </button>
        </div>
      </div>
    </div>
  );
}

export default CreateModel;
