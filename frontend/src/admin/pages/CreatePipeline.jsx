import React, { useState } from "react";
import AdminNavbar from "../../components/AdminNavbar";
import uploadImage from "../../assets/upload.png";
import bulletImage from "../../assets/bullete.img.png";

const CreatePipeline = () => {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [selectedOptions, setSelectedOptions] = useState([]); // Tracks selected models
  const models = ["Model 1", "Model 2", "Model 3", "Model 4"]; // Available models

  const toggleDropdown = () => {
    setIsDropdownOpen(!isDropdownOpen);
  };

  const handleOptionChange = (model) => {
    if (selectedOptions.includes(model)) {
      setSelectedOptions(selectedOptions.filter((item) => item !== model));
    } else {
      setSelectedOptions([...selectedOptions, model]);
    }
  };

  const handleSelectAll = () => {
    if (selectedOptions.length === models.length) {
      setSelectedOptions([]); // Deselect all if everything is selected
    } else {
      setSelectedOptions(models); // Select all options
    }
  };

  return (
    <div className="bg-[#EAECFF] min-h-screen">
      {/* Navbar */}
      <AdminNavbar />

      {/* Content */}
      <div className="container mt-7 mx-auto p-8">
        <h1 className="text-3xl font-semibold text-center text-[#39407D]">
          Create Pipeline
        </h1>

        {/* Form Section */}
        <div className="mt-8 rounded-lg p-8 max-w-7xl mx-auto">
          {/* Pipeline Name */}
          <input
            type="text"
            placeholder="Enter name of Pipeline"
            className="w-full border border-gray-300 rounded-full px-4 py-2 mb-6 text-[#6966FF] focus:outline-none focus:ring-2 focus:ring-[#6966FF]"
          />

          {/* Model Selection and File Upload Section */}
          <div className="flex item-center justify-center space-x-6">
            {/* Model Selection */}
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
                <i
                  className={`material-icons ${
                    isDropdownOpen ? "rotate-180" : ""
                  }`}
                >
                  ▲
                </i>
              </div>

              {isDropdownOpen && (
                <div className="absolute w-full mt-2 bg-white rounded-lg shadow-lg z-10 max-h-60 overflow-auto">
                  {/* Select/Deselect All */}
                  <div
                    className="px-4 py-2 text-[#6966FF] hover:bg-[#EAECFF] cursor-pointer"
                    onClick={handleSelectAll}
                  >
                    {selectedOptions.length === models.length
                      ? "Deselect All"
                      : "Select All"}
                  </div>
                  {/* Dropdown Options */}
                  {models.map((model) => (
                    <div
                      key={model}
                      className="px-4 py-2 text-[#6966FF] hover:bg-[#EAECFF] cursor-pointer flex items-center"
                      onClick={() => handleOptionChange(model)}
                    >
                      <input
                        type="checkbox"
                        checked={selectedOptions.includes(model)}
                        onChange={() => handleOptionChange(model)}
                        className="mr-2"
                      />
                      {/* Custom Bullet Image */}
                      <img
                        src={bulletImage}
                        alt="Bullet"
                        className="w-4 h-4 mr-2"
                      />
                      {model}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* File Upload */}
            {/* File Upload */}
            <div className="flex-1 h-[150px] bg-white rounded-[40px] shadow-md p-4 flex items-center justify-center">
              <label
                htmlFor="file-upload"
                className="w-full h-full bg-[#EAECFF] text-[#6966FF] p-4 rounded-full flex flex-col items-center justify-center cursor-pointer"
              >
                {/* Upload Image */}
                <img
                  src={uploadImage}
                  alt="Upload"
                  className="mb-2 w-12 h-12"
                />
                {/* Upload Text */}
                <span>Upload your source File</span>
              </label>
              <input
                id="file-upload"
                type="file"
                className="hidden"
                onChange={(e) => {
                  console.log("File selected:", e.target.files[0]); // Handle file here
                }}
              />
            </div>
          </div>

          {/* Submit Button */}
          <div className="mt-8 flex justify-center">
            <button className="bg-[#6966FF] text-white px-8 py-3 mt-20 rounded-full shadow-lg text-lg">
              Create
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CreatePipeline;
