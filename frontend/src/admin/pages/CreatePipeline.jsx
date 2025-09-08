import React, { useEffect, useState } from "react";
import AdminNavbar from "../../components/AdminNavbar";
import uploadImage from "../../assets/upload.png";
import bulletImage from "../../assets/bullete.img.png";

const CreatePipeline = () => {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [models, setModels] = useState([]);
  const [selectedOptions, setSelectedOptions] = useState([]);
  const [pipelineName, setPipelineName] = useState("");   // ✅ NEW: store pipeline name

  // 🔹 Fetch models from backend
  useEffect(() => {
    fetch("http://localhost:8000/model/list/")
      .then((res) => res.json())
      .then((data) => setModels(data.models))
      .catch((err) => console.error("Error fetching models:", err));
  }, []);

  const toggleDropdown = () => setIsDropdownOpen(!isDropdownOpen);

  const handleOptionChange = (modelName) => {
    setSelectedOptions((prev) => {
      if (prev.includes(modelName)) {
        return prev.filter((m) => m !== modelName);
      } else {
        return [...prev, modelName];
      }
    });
  };

  const handleSelectAll = () => {
    if (selectedOptions.length === models.length) {
      setSelectedOptions([]);
    } else {
      setSelectedOptions(models.map((m) => m.name));
    }
  };

  // ✅ NEW: POST request to save pipeline
  const handleCreatePipeline = () => {
    if (!pipelineName.trim()) {
      alert("Please enter a pipeline name.");
      return;
    }
    if (selectedOptions.length === 0) {
      alert("Please select at least one model.");
      return;
    }

    fetch("http://localhost:8000/model/pipelines/create/", {   // ✅ Your Django route
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
      name: pipelineName,
      is_active: true,
      allowed_models: selectedOptions,
      created_by: "admin"   // ✅ send username
    }),
    })
      .then((res) => res.json())
      .then((data) => {
        console.log("Pipeline created:", data);
        alert("✅ Pipeline created successfully!");
        setPipelineName("");
        setSelectedOptions([]);
      })
      .catch((err) => {
        console.error("Error creating pipeline:", err);
        alert("❌ Failed to create pipeline");
      });
  };

  return (
    <div className="bg-[#EAECFF] min-h-screen">
      <AdminNavbar />

      <div className="container mt-7 mx-auto p-8">
        <h1 className="text-3xl font-semibold text-center text-[#39407D]">
          Create Pipeline
        </h1>

        <div className="mt-8 rounded-lg p-8 max-w-7xl mx-auto">
          {/* ✅ Pipeline Name */}
          <input
            type="text"
            placeholder="Enter name of Pipeline"
            value={pipelineName}
            onChange={(e) => setPipelineName(e.target.value)}
            className="w-full border border-gray-300 rounded-full px-4 py-2 mb-6 text-[#6966FF] focus:outline-none focus:ring-2 focus:ring-[#6966FF]"
          />

          {/* ✅ Dropdown & File Upload section stays same */}
          <div className="flex item-center justify-center space-x-6">
            {/* Dropdown */}
            <div className="relative flex-1 h-[150px] w-1/2 bg-white rounded-[40px] shadow-md p-4">
              <h1 className="mt-4 mb-4 font-thin text-center">Select Models</h1>
              <div
                className="w-full border border-gray-300 rounded-full px-4 py-2 flex justify-between items-center cursor-pointer bg-[#6966FF] text-white"
                onClick={toggleDropdown}
              >
                <span>
                  {selectedOptions.length === 0
                    ? "Choose Models"
                    : `${selectedOptions.length} Model(s) Selected`}
                </span>
                <i className={`material-icons ${isDropdownOpen ? "rotate-180" : ""}`}>
                  ▲
                </i>
              </div>

              {isDropdownOpen && (
                <div className="absolute w-full mt-2 bg-white rounded-lg shadow-lg z-10 max-h-60 overflow-auto">
                  <div
                    className="px-4 py-2 text-[#6966FF] hover:bg-[#EAECFF] cursor-pointer"
                    onClick={handleSelectAll}
                  >
                    {selectedOptions.length === models.length ? "Deselect All" : "Select All"}
                  </div>
                  {models.map((model) => (
                    <div
                      key={model.id}
                      className="px-4 py-2 text-[#6966FF] hover:bg-[#EAECFF] cursor-pointer flex items-center"
                      onClick={() => handleOptionChange(model.name)}
                    >
                      <input
                        type="checkbox"
                        checked={selectedOptions.includes(model.name)}
                        onChange={() => handleOptionChange(model.name)}
                        className="mr-2"
                      />
                      <img src={bulletImage} alt="Bullet" className="w-4 h-4 mr-2" />
                      {model.name}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* File Upload (unchanged) */}
            <div className="flex-1 h-[150px] bg-white rounded-[40px] shadow-md p-4 flex items-center justify-center">
              <label
                htmlFor="file-upload"
                className="w-full h-full bg-[#EAECFF] text-[#6966FF] p-4 rounded-full flex flex-col items-center justify-center cursor-pointer"
              >
                <img src={uploadImage} alt="Upload" className="mb-2 w-12 h-12" />
                <span>Upload your source File</span>
              </label>
              <input id="file-upload" type="file" className="hidden" />
            </div>
          </div>

          {/* ✅ Submit Button */}
          <div className="mt-8 flex justify-center">
            <button
              className="bg-[#6966FF] text-white px-8 py-3 mt-20 rounded-full shadow-lg text-lg"
              onClick={handleCreatePipeline}
            >
              Create
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CreatePipeline;
