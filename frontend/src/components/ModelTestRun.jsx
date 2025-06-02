import React, { useState } from "react";
import "../styles/customStyles.css";
import { useNavigate } from "react-router-dom";

const ModelTestRun = () => {
  const allModels = ["MODEL 1", "MODEL 2", "MODEL 3", "MODEL 4", "MODEL 5"];
  const allClasses = ["class 1", "class 2", "class 3", "class 4"];
  const algorithms = [
    "Algorithm 1",
    "Algorithm 2",
    "Algorithm 3",
    "Algorithm 4",
  ];

  const [selectedModels, setSelectedModels] = useState([]);
  const [classesOpen, setClassesOpen] = useState(false);
  const [selectedClasses, setSelectedClasses] = useState(allClasses);
  const [algoOpen, setAlgoOpen] = useState(false);
  const [selectedAlgorithms, setSelectedAlgorithms] = useState(algorithms);

  const navigate = useNavigate();

  const toggleModel = (model) => {
    setSelectedModels((prev) =>
      prev.includes(model) ? prev.filter((m) => m !== model) : [...prev, model]
    );
  };

  const toggleClass = (className) => {
    setSelectedClasses((prev) =>
      prev.includes(className)
        ? prev.filter((c) => c !== className)
        : [...prev, className]
    );
  };

  const toggleAlgorithm = (algo) => {
    setSelectedAlgorithms((prev) =>
      prev.includes(algo)
        ? prev.filter((a) => a !== algo)
        : [...prev, algo]
    );
  };

  // Function to handle "Select All/Deselect All" for classes
  const toggleAllClasses = () => {
    if (selectedClasses.length === allClasses.length) {
      setSelectedClasses([]); // Deselect all if all are selected
    } else {
      setSelectedClasses(allClasses); // Select all
    }
  };

  // Function to handle "Select All/Deselect All" for algorithms
  const toggleAllAlgorithms = () => {
    if (selectedAlgorithms.length === algorithms.length) {
      setSelectedAlgorithms([]); // Deselect all if all are selected
    } else {
      setSelectedAlgorithms(algorithms); // Select all
    }
  };

  const goBack = () => {
    navigate("/model-test");
  };

  return (
    <div className="bg-[#EAECFF] min-h-[668px] w-full flex items-center justify-center p-6">
      <div className="bg-white shadow-lg rounded-[20px] w-full max-w-5xl overflow-hidden relative p-10">
        <div className="flex flex-col h-full space-y-4">
          <div className="flex-grow space-x-6 flex mb-6 justify-between">
            {/* Selected Models */}
            <div className="w-1/4 bg-[#6966FF] rounded-[15px] p-4 text-white flex flex-col">
              <h2 className="text-xl font-bold mb-4 text-center">
                Selected Models
              </h2>
              <div className="h-[1px] bg-white w-full mb-5"></div>
              <div className="overflow-y-auto flex-grow">
                {allModels.map((model, index) => (
                  <div
                    key={index}
                    className="flex items-center mb-2 justify-center"
                  >
                    <input
                      type="checkbox"
                      id={`model-${index}`}
                      checked={selectedModels.includes(model)}
                      onChange={() => toggleModel(model)}
                      className="custom-checkbox mr-2"
                    />
                    <label htmlFor={`model-${index}`} className="text-center">
                      {model}
                    </label>
                  </div>
                ))}
              </div>
            </div>

            {/* Right side content */}
            <div className="w-3/4 flex flex-col justify-between relative">
              <div className="space-y-4">
                <div className="flex justify-between">
                  {/* Select Class Dropdown */}
                  <div className="w-5/12 relative z-20">
                    <button
                      onClick={() => setClassesOpen(!classesOpen)}
                      className="bg-[#6966FF] text-white py-2 px-4 rounded-full w-full text-left flex justify-between items-center"
                    >
                      Select Class
                      <span>{classesOpen ? "▲" : "▼"}</span>
                    </button>
                    {classesOpen && (
                      <div className="absolute z-50 w-full bg-[#6966FF] border border-gray-300 mt-1 rounded-lg shadow-lg">
                        <div className="max-h-48 overflow-y-auto">
                          {allClasses.map((cls, index) => (
                            <div
                              key={index}
                              className="flex items-center p-2 hover:bg-gray-200 hover:text-black text-white"
                            >
                              <input
                                type="checkbox"
                                id={`class-${index}`}
                                checked={selectedClasses.includes(cls)}
                                onChange={() => toggleClass(cls)}
                                className="custom-checkbox mr-2"
                              />
                              <label htmlFor={`class-${index}`}>{cls}</label>
                            </div>
                          ))}
                        </div>
                        <div className="flex justify-end p-2">
                          <button
                            className="bg-[#7A8BFF] text-white py-1 px-3 rounded-full"
                            onClick={toggleAllClasses}
                          >
                            {selectedClasses.length === allClasses.length
                              ? "Deselect All"
                              : "Select All"}
                          </button>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Select XAI Algo Dropdown */}
                  <div className="w-5/12 relative z-20">
                    <button
                      onClick={() => setAlgoOpen(!algoOpen)}
                      className="bg-[#6966FF] text-white py-2 px-4 rounded-full w-full text-left flex justify-between items-center"
                    >
                      Select XAI Algo
                      <span>{algoOpen ? "▲" : "▼"}</span>
                    </button>
                    {algoOpen && (
                      <div className="absolute z-50 w-full bg-[#6966FF] border border-gray-300 mt-1 rounded-lg shadow-lg">
                        <div className="max-h-48 overflow-y-auto">
                          {algorithms.map((algo, index) => (
                            <div
                              key={index}
                              className="flex items-center p-2 hover:bg-gray-200 hover:text-black text-white"
                            >
                              <input
                                type="checkbox"
                                id={`algo-${index}`}
                                checked={selectedAlgorithms.includes(algo)}
                                onChange={() => toggleAlgorithm(algo)}
                                className="custom-checkbox mr-2"
                              />
                              <label htmlFor={`algo-${index}`}>{algo}</label>
                            </div>
                          ))}
                        </div>
                        <div className="flex justify-end p-2">
                          <button
                            className="bg-[#7A8BFF] text-white py-1 px-3 rounded-full"
                            onClick={toggleAllAlgorithms}
                          >
                            {selectedAlgorithms.length === algorithms.length
                              ? "Deselect All"
                              : "Select All"}
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Buttons */}
              <div className="flex justify-between mt-4">
                <button className="bg-[#6966FF] text-white py-2 px-4 rounded-full flex items-center justify-center w-full mr-2">
                  <svg
                    className="w-5 h-5 mr-2"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
                    />
                  </svg>
                  Upload Test Image
                </button>

                <button className="bg-[#6966FF] text-white py-2 px-4 rounded-full flex items-center justify-center w-full">
                  Run Instance
                </button>
              </div>
            </div>
          </div>

          {/* Back Button */}
          <div className="flex justify-end mt-4">
            <button
              className="bg-[#6966FF] text-white py-2 px-10 rounded-full"
              onClick={goBack}
            >
              Back
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelTestRun;
