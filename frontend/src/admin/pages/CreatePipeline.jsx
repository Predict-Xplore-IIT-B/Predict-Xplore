import React, { useEffect, useState, useRef } from "react";
import { useSelector } from "react-redux";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { toast } from "react-toastify";
import AdminNavbar from "../../components/AdminNavbar";
import uploadImage from "../../assets/upload.png";
import bulletImage from "../../assets/bullete.img.png";

const CreatePipeline = () => {
  const navigate = useNavigate();

  const user = useSelector((state) => state.user.users[state.user.users.length - 1]);
  const token = user?.token;

  const [imgUpload, setImgUpload] = useState(false);
  const [testCaseId, setTestCaseId] = useState(null);
  const [pipelineName, setPipelineName] = useState("");
  const [models, setModels] = useState([]);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  // selectedModels: ordered array of objects { id, name }
  const [selectedModels, setSelectedModels] = useState([]);
  const [reportUrl, setReportUrl] = useState(null);
  const draggingIndex = useRef(null);
  const dropdownRef = useRef(null);

  // Step 1: Fetch Available Models on mount
  const retrieveModels = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:8000/model/list/");
      // expecting response.data.models as array
      setModels(response.data.models || response.data || []);
    } catch (e) {
      console.error("Error fetching models", e);
      toast.error("Error fetching models");
    }
  };

  useEffect(() => {
    retrieveModels();
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleOutsideClick = (e) => {
      if (isDropdownOpen && dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener("mousedown", handleOutsideClick);
    return () => document.removeEventListener("mousedown", handleOutsideClick);
  }, [isDropdownOpen]);

  // Dropdown helpers
  const toggleDropdown = () => setIsDropdownOpen((s) => !s);

  const handleSelectAll = () => {
    if (selectedModels.length === models.length) {
      setSelectedModels([]);
    } else {
      // keep order from models list
      const all = models.map((m) => ({ id: m.id, name: m.name }));
      setSelectedModels(all);
    }
  };

  const handleOptionChange = (model) => {
    // model may be object or name/id; accept object
    const id = model.id ?? model;
    const name = model.name ?? model.name;
    const exists = selectedModels.find((m) => m.id === id);
    if (exists) {
      setSelectedModels((prev) => prev.filter((m) => m.id !== id));
    } else {
      // append to end
      const modelObj = models.find((m) => m.id === id) || (typeof model === "object" ? model : { id, name: model });
      setSelectedModels((prev) => [...prev, { id: modelObj.id, name: modelObj.name }]);
    }
  };

  // Drag and drop handlers for ordering selectedModels
  const onDragStart = (e, index) => {
    draggingIndex.current = index;
    e.dataTransfer.effectAllowed = "move";
  };

  const onDragOver = (e, index) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  };

  const onDrop = (e, dropIndex) => {
    e.preventDefault();
    const dragIdx = draggingIndex.current;
    if (dragIdx === null || dragIdx === undefined) return;
    if (dragIdx === dropIndex) return;
    setSelectedModels((prev) => {
      const copy = [...prev];
      const [moved] = copy.splice(dragIdx, 1);
      copy.splice(dropIndex, 0, moved);
      return copy;
    });
    draggingIndex.current = null;
  };

  // Step 2: Upload the Source Image
  const uploadTestImage = async (file) => {
    const formData = new FormData();
    formData.append("image", file);
    setImgUpload(true);

    try {
      const imgResponse = await axios.post(
        "http://127.0.0.1:8000/model/instance/upload",
        formData,
        {
          headers: {
            Authorization: token ? `Token ${token}` : "",
            // let browser set Content-Type with boundary
          },
        }
      );

      if ((imgResponse.status === 200 || imgResponse.status === 201) && imgResponse.data?.test_case_id) {
        setTestCaseId(imgResponse.data.test_case_id);
        toast.success("Image uploaded successfully", { autoClose: 2000 });
      } else {
        toast.error("Unexpected response from image upload", { autoClose: 2000 });
      }
    } catch (e) {
      console.error("Error in uploading image", e);
      toast.error("Error in uploading image", { autoClose: 2000 });
    } finally {
      setImgUpload(false);
    }
  };

  // Step 3: Execute the Pipeline
  const executePipeline = async () => {
    if (!testCaseId) return toast.error("Upload an image first");
    if (!pipelineName) return toast.error("Please enter a pipeline name");
    if (selectedModels.length === 0) return toast.error("Select at least one model");

    const modelIds = selectedModels.map((m) => m.id);

    try {
      const resp = await axios.post(
        "http://127.0.0.1:8000/model/pipelines/predict",
        {
          test_case_id: testCaseId,
          models: modelIds,
          pipeline_name: pipelineName,
        },
        {
          headers: {
            Authorization: token ? `Token ${token}` : "",
            "Content-Type": "application/json",
          },
        }
      );

      if (resp.status === 200 || resp.status === 201) {
        toast.success("Pipeline executed. Report generated.");
        // try to read download_url from response
        const reports = resp.data?.reports;
        if (reports && reports.length > 0 && reports[0].download_url) {
          setReportUrl(reports[0].download_url);
        }
      } else {
        toast.error("Pipeline execution failed");
      }
    } catch (e) {
      console.error("Pipeline call failed", e);
      toast.error("Pipeline execution failed: " + (e?.response?.data?.message || e.message));
    }
  };

  return (
    <div className="bg-[#EAECFF] min-h-screen">
      <AdminNavbar />

      <div className="container mt-7 mx-auto p-8">
        <h1 className="text-3xl font-semibold text-center text-[#39407D]">
          Create Pipeline
        </h1>

        <div className="mt-8 rounded-lg p-8 max-w-7xl mx-auto">
          {/* Pipeline Name */}
          <input
            type="text"
            placeholder="Enter name of Pipeline"
            value={pipelineName}
            onChange={(e) => setPipelineName(e.target.value)}
            className="w-full border border-gray-300 rounded-full px-4 py-2 mb-6 text-[#6966FF] focus:outline-none focus:ring-2 focus:ring-[#6966FF]"
          />

          <div className="flex item-center justify-center space-x-6">
            {/* Dropdown / model list */}
            <div ref={dropdownRef} className="relative flex-1 w-1/2 bg-white rounded-[24px] shadow-md p-4 flex flex-col">
              <h1 className="mt-2 mb-4 font-thin text-center">Select Models</h1>
              <div
                className="w-full border border-gray-300 rounded-full px-4 py-2 flex justify-between items-center cursor-pointer bg-[#6966FF] text-white"
                onClick={toggleDropdown}
              >
                <span>
                  {selectedModels.length === 0
                    ? "Choose Models"
                    : `${selectedModels.length} Model(s) Selected`}
                </span>
                <i className={`material-icons ${isDropdownOpen ? "rotate-180" : ""}`}>
                  ▲
                </i>
              </div>

              {isDropdownOpen && (
                <div className="absolute left-4 right-4 mt-4 bg-white rounded-lg shadow-lg z-10 max-h-60 overflow-auto">
                  <div
                    className="px-4 py-2 text-[#6966FF] hover:bg-[#EAECFF] cursor-pointer"
                    onClick={handleSelectAll}
                  >
                    {selectedModels.length === models.length ? "Deselect All" : "Select All"}
                  </div>
                  {models.map((model) => (
                    <label
                      key={model.id}
                      className="flex items-center px-4 py-2 text-[#6966FF] hover:bg-[#EAECFF] cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={!!selectedModels.find((m) => m.id === model.id)}
                        onChange={() => handleOptionChange(model)}
                        className="mr-2"
                      />
                      <img src={bulletImage} alt="Bullet" className="w-4 h-4 mr-2" />
                      <span>{model.name}</span>
                    </label>
                  ))}
                </div>
              )}

              {/* Selected models (draggable order) */}
              <div className="mt-4 flex-1 flex flex-col">
                <h2 className="text-sm mb-2 text-gray-600">Ordered Models (drag to reorder)</h2>
                <div className="mt-2 space-y-2 overflow-auto">
                  {selectedModels.length === 0 && (
                    <div className="px-4 py-3 text-gray-400">No models selected</div>
                  )}
                  {selectedModels.map((m, idx) => (
                    <div
                      key={m.id}
                      draggable
                      onDragStart={(e) => onDragStart(e, idx)}
                      onDragOver={(e) => onDragOver(e, idx)}
                      onDrop={(e) => onDrop(e, idx)}
                      className="flex items-center justify-between p-3 bg-[#F7F8FF] rounded-md border"
                    >
                      <div className="flex items-center">
                        <div className="mr-3 text-xs text-gray-500">{idx + 1}</div>
                        <div className="font-medium text-[#6966FF]">{m.name}</div>
                      </div>
                      <button
                        className="text-red-500 text-sm"
                        onClick={() => setSelectedModels((prev) => prev.filter((x) => x.id !== m.id))}
                      >
                        Remove
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* File Upload */}
            <div className="flex-1 bg-white rounded-[24px] shadow-md p-6 flex flex-col items-center justify-start min-h-[180px]">
              <img src={uploadImage} alt="upload" className="w-24 h-24 mb-4" />
              <label className={`${imgUpload ? "bg-gray-400" : "bg-[#6966FF]"} text-white py-2 px-4 rounded-full w-[80%] text-center cursor-pointer`}>
                {imgUpload ? "Image Uploading ..." : (testCaseId ? "Image Uploaded" : "Upload Test Image")}
                <input
                  type="file"
                  accept=".jpg,.jpeg,.png"
                  className="hidden"
                  disabled={imgUpload}
                  onChange={(e) => {
                    const file = e.target.files[0];
                    if (file) uploadTestImage(file);
                  }}
                />
              </label>
              {testCaseId && <div className="mt-4 text-sm text-green-600">Uploaded: test_case_id {testCaseId}</div>}
            </div>
          </div>

          {/* Submit Button & download link */}
          <div className="mt-8 flex flex-col items-center">
            <button
              className={`bg-[#6966FF] text-white px-8 py-3 mt-6 rounded-full shadow-lg text-lg ${(!testCaseId || selectedModels.length === 0 || !pipelineName) ? "opacity-60 cursor-not-allowed" : ""}`}
              onClick={executePipeline}
              disabled={!testCaseId || selectedModels.length === 0 || !pipelineName}
            >
              Create
            </button>

            {reportUrl && (
              <a href={reportUrl} target="_blank" rel="noreferrer" className="mt-4 text-sm text-blue-600 underline">
                Download Report
              </a>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CreatePipeline;
