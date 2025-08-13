import React, { useState } from "react";
import AdminNavbar from "../../components/AdminNavbar";
import SelectCheckbox from "../../components/SelectCheckbox";
import upload from "../../assets/upload.png";
import axios from "axios";
import { toast } from "react-toastify";
import SingleSelectDropdown from "../../components/SingleSelectDropdown";
import { useSelector } from "react-redux";

function CreateModel() {
  const user = useSelector((state) => state.user.users[state.user.users.length -1]);

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [modelFile, setModelFile] = useState(null);
  const [modelImage, setModelImage] = useState(null);
  const [modelType, setModelType] = useState("");
  const [createdBy, setCreatedBy] = useState(""); // hardcoded or fetch from redux/auth

  const [selectedClass, setSelectedClass] = useState([]);
  const [selectedRoles, setSelectedRoles] = useState([]);
  const [isFileUploaded, setIsFileUploaded] = useState(false);
  const [isImageUploaded, setIsImageUploaded] = useState(false);

  const classes = ["class 1", "class 5", "class 3", "class 4"];
  const roles = ["CSE DEPT", "AIML DEPT", "MECH DEPT", "DS DEPT"];
  const types = [
  { label: "Human Detection", value: "HumanDetection" },
  { label: "Image Segmentation", value: "ImageSegmentation" },
  { label: "Object Detection", value: "ObjectDetection" },
  // { label: "Classification", value: "Classification" },
  // { label: "Decision Tree", value: "DecisionTree" },
];


  const handleUpload = async () => {
    if (!user || !user.token) {
      alert("User or session token not found!");
      return;
    }

    const username = user.username;
    const token = user.token;

    console.log("Uploading model with details:", {
      name,
      description,
      modelFile,
      modelImage,
      modelType,
      createdBy: username,
      selectedClass,
      selectedRoles
    });

    if (!name || !description || !modelFile || !modelImage || !modelType || !username) {
      alert("All fields are required!");
      return;
    }

    const formData = new FormData();
    formData.append("name", name);
    formData.append("description", description);
    formData.append("model_file", modelFile);
    formData.append("model_image", modelImage);
    formData.append("model_type", modelType);
    formData.append("created_by", username);
    // Add selected classes and roles as JSON strings
    formData.append("classes", JSON.stringify(selectedClass));
    formData.append("roles", JSON.stringify(selectedRoles));

    try {
      const response = await axios.post("http://127.0.0.1:8000/model/create", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          "Authorization": `Token ${token}`,
        },
      });
      console.log("Upload success:", response.data);
      toast.success("Model successfully uploaded!", { autoClose: 2000 });
    } catch (error) {
      console.error("Error uploading model:", error);
      toast.error("Failed to upload model.");
    }
  };


  return (
    <div className="h-screen w-screen bg-[#EAECFF] overflow-auto">
      <AdminNavbar />
      <div className="flex flex-col mx-20">
        <div>
          <input
            type="text"
            className="w-[35%] rounded-full border px-3 py-2 mt-24"
            placeholder="Enter name of the model"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </div>

        <div className="flex justify-end mt-5 gap-4">
          <SelectCheckbox options={classes} title="Select Class" setSelected={setSelectedClass} />
          <SelectCheckbox options={roles} title="Select Roles" setSelected={setSelectedRoles} />
          <SingleSelectDropdown options={types} title="Select Model Type" setSelected={setModelType} />

        </div>

        <div className="mt-4">
          <textarea
            className="w-full h-[9vh] p-2 rounded-2xl resize-none"
            placeholder="Enter Description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
        </div>

        <div className="mt-5">
          <div className="flex items-center justify-center w-full h-[25vh] bg-white rounded-2xl">
            <div className="m-8 pr-20">
              <label htmlFor="model-file" className="cursor-pointer hover:bg-gray-200">
                <div className="flex flex-col items-center justify-center">
                  {isFileUploaded ? (
                    <svg className="h-20 w-20 animate-pulse text-green-600" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    <img src={upload} alt="Upload model file" className="h-20" />
                  )}
                </div>
                <input
                  id="model-file"
                  type="file"
                  className="hidden"
                  onChange={(e) => {
                    setModelFile(e.target.files[0]);
                    setIsFileUploaded(true);
                  }}
                />
              </label>
              <p className="my-2 text-xl text-gray-500">Upload your Model here</p>
            </div>

            <div className="m-8 pl-20">
              <label htmlFor="model-image" className="cursor-pointer hover:bg-gray-200">
                <div className="flex flex-col items-center justify-center">
                  {isImageUploaded ? (
                    <svg className="h-20 w-20 animate-pulse text-green-600" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    <img src={upload} alt="Upload model thumbnail image" className="h-20" />
                  )}
                </div>
                <input
                  id="model-image"
                  type="file"
                  className="hidden"
                  onChange={(e) => {
                    setModelImage(e.target.files[0]);
                    setIsImageUploaded(true);
                  }}
                />
              </label>
              <p className="my-2 text-xl text-gray-500">Upload Model thumbnail Image</p>
            </div>
          </div>
        </div>

        <div className="w-full mt-6 flex justify-center items-center">
          <button
            type="button"
            onClick={handleUpload}
            className="text-white bg-[#6966FF] hover:bg-blue-800 focus:outline-none focus:ring-4 focus:ring-blue-300 font-extrabold rounded-full text-2xl px-12 py-2.5 text-center"
          >
            Create
          </button>
        </div>
      </div>
    </div>
  );
}

export default CreateModel;
