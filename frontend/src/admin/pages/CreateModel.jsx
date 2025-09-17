import React, { useState } from "react";
import AdminNavbar from "../../components/AdminNavbar";
import SelectCheckbox from "../../components/SelectCheckbox";
import upload from "../../assets/upload.png";
import axios from "axios";
import { toast } from "react-toastify";
import SingleSelectDropdown from "../../components/SingleSelectDropdown";
import { useSelector } from "react-redux";

function CreateModel() {
  const user = useSelector(
    (state) => state.user.users[state.user.users.length - 1]
  );

  const [uploadMode, setUploadMode] = useState("standard"); // "standard" | "containerized"

  // Common fields
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  // Standard mode fields
  const [modelFile, setModelFile] = useState(null);
  const [modelImage, setModelImage] = useState(null);
  const [modelType, setModelType] = useState("");
  const [selectedClass, setSelectedClass] = useState([]);
  const [selectedRoles, setSelectedRoles] = useState([]);
  const [isFileUploaded, setIsFileUploaded] = useState(false);
  const [isImageUploaded, setIsImageUploaded] = useState(false);

  // Containerized mode fields
  const [zipFile, setZipFile] = useState(null);
  const [isZipUploaded, setIsZipUploaded] = useState(false);
  const [allowedUsers, setAllowedUsers] = useState([]);

  const classes = ["class 1", "class 5", "class 3", "class 4"];
  const roles = ["CSE DEPT", "AIML DEPT", "MECH DEPT", "DS DEPT"];
  const types = [
    { label: "Human Detection", value: "HumanDetection" },
    { label: "Image Segmentation", value: "ImageSegmentation" },
    { label: "Object Detection", value: "ObjectDetection" },
  ];

  const handleUpload = async () => {
    if (!user || !user.token) {
      alert("User or session token not found!");
      return;
    }

    const token = user.token;

    try {
      let formData = new FormData();

      if (uploadMode === "standard") {
        if (!name || !description || !modelFile || !modelImage || !modelType) {
          alert("All fields are required!");
          return;
        }

        formData.append("name", name);
        formData.append("description", description);
        formData.append("model_file", modelFile);
        formData.append("model_image", modelImage);
        formData.append("model_type", modelType);
        formData.append("created_by", user.username);
        formData.append("classes", JSON.stringify(selectedClass));
        formData.append("roles", JSON.stringify(selectedRoles));

        const response = await axios.post(
          "http://127.0.0.1:8000/model/create",
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
              Authorization: `Token ${token}`,
            },
          }
        );
        toast.success("Model successfully uploaded!", { autoClose: 2000 });
        console.log("Standard upload:", response.data);
      } else {
        if (!name || !description || !zipFile) {
          alert("All fields are required!");
          return;
        }

        formData.append("name", name);
        formData.append("description", description);
        formData.append("zipfile", zipFile);
        formData.append("allowed_users", JSON.stringify(allowedUsers));

        const response = await axios.post(
          "http://127.0.0.1:8000/model/create-container/",
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
              Authorization: `Token ${token}`,
            },
          }
        );
        toast.success("Container created successfully!", { autoClose: 2000 });
        console.log("Containerized upload:", response.data);
      }
    } catch (error) {
      console.error("Error uploading:", error);
      toast.error("Upload failed.");
    }
  };

  return (
    <div className="h-screen w-screen bg-[#EAECFF] overflow-auto">
      <AdminNavbar />
      <div className="flex flex-col mx-20">
        {/* Upload Mode Toggle */}
        <div className="mt-6">
          <label className="mr-4 font-bold">Upload Mode:</label>
          <select
            value={uploadMode}
            onChange={(e) => setUploadMode(e.target.value)}
            className="border rounded px-3 py-2"
          >
            <option value="standard">Standard</option>
            <option value="containerized">Containerized (ZIP)</option>
          </select>
        </div>

        {/* Common Inputs */}
        <div className="mt-6">
          <input
            type="text"
            className="w-[35%] rounded-full border px-3 py-2"
            placeholder="Enter model name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <textarea
            className="w-full h-[9vh] p-2 mt-4 rounded-2xl resize-none"
            placeholder="Enter description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
        </div>

        {uploadMode === "standard" ? (
          <>
            {/* Classes, Roles, Types */}
            <div className="flex justify-end mt-5 gap-4">
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
              <SingleSelectDropdown
                options={types}
                title="Select Model Type"
                setSelected={setModelType}
              />
            </div>

            {/* File Uploads */}
            <div className="mt-5 flex items-center justify-center w-full h-[25vh] bg-white rounded-2xl">
              {/* Model File */}
              <div className="m-8 pr-20">
                <label htmlFor="model-file" className="cursor-pointer">
                  <div className="flex flex-col items-center">
                    {isFileUploaded ? (
                      <svg
                        className="h-20 w-20 animate-pulse text-green-600"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                    ) : (
                      <img src={upload} alt="Upload model" className="h-20" />
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
                <p className="my-2 text-xl text-gray-500">
                  Upload your Model here
                </p>
              </div>

              {/* Thumbnail Image */}
              <div className="m-8 pl-20">
                <label htmlFor="model-image" className="cursor-pointer">
                  <div className="flex flex-col items-center">
                    {isImageUploaded ? (
                      <svg
                        className="h-20 w-20 animate-pulse text-green-600"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                    ) : (
                      <img src={upload} alt="Upload image" className="h-20" />
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
                <p className="my-2 text-xl text-gray-500">
                  Upload Model Thumbnail
                </p>
              </div>
            </div>
          </>
        ) : (
          <>
            {/* Containerized Upload (ZIP) */}
            <div className="mt-5 flex items-center justify-center w-full h-[20vh] bg-white rounded-2xl">
              <div className="m-8">
                <label htmlFor="zip-file" className="cursor-pointer">
                  <div className="flex flex-col items-center">
                    {isZipUploaded ? (
                      <svg
                        className="h-20 w-20 animate-pulse text-green-600"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                    ) : (
                      <img src={upload} alt="Upload zip" className="h-20" />
                    )}
                  </div>
                  <input
                    id="zip-file"
                    type="file"
                    className="hidden"
                    accept=".zip"
                    onChange={(e) => {
                      setZipFile(e.target.files[0]);
                      setIsZipUploaded(true);
                    }}
                  />
                </label>
                <p className="my-2 text-xl text-gray-500">
                  Upload ZIP for containerization
                </p>
              </div>
            </div>

            {/* Allowed Users */}
            <div className="mt-5">
              <SelectCheckbox
                options={["user1", "user2", "user3"]}
                title="Allowed Users"
                setSelected={setAllowedUsers}
              />
            </div>
          </>
        )}

        {/* Submit Button */}
        <div className="w-full mt-6 flex justify-center items-center">
          <button
            type="button"
            onClick={handleUpload}
            className="text-white bg-[#6966FF] hover:bg-blue-800 font-extrabold rounded-full text-2xl px-12 py-2.5"
          >
            {uploadMode === "standard" ? "Create Model" : "Create Container"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default CreateModel;
