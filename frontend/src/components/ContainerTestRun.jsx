import React, { useState } from "react";
import { useSelector, useDispatch } from "react-redux";
import { toggleContainerToRun } from "../redux/reducers/containerSlice";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { toast } from "react-toastify";
import AdminNavbar from "../components/AdminNavbar";

function ContainerTestRun() {
  const dispatch = useDispatch();
  const navigate = useNavigate();

  const user = useSelector(
    (state) => state.user.users[state.user.users.length - 1]
  );

  const selectedContainers = useSelector(
    (state) => state.containers?.containers.filter((c) => c.selected) || []
  );
  const toRunContainers = useSelector(
    (state) => state.containers?.containers.filter((c) => c.toRun) || []
  );

  const [modalData, setModalData] = useState(null); // Holds video URL
  const [modalTitle, setModalTitle] = useState(""); // Container name
  const [results, setResults] = useState([]);
  const [fileUploading, setFileUploading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [running, setRunning] = useState(false);

  const toggleContainer = (container) => {
    dispatch(toggleContainerToRun(container.id));
  };

  const goBack = () => {
    navigate("/model-test");
  };

  const handleFileUpload = (file) => {
    if (!file) return;
    setUploadedFile(file);
    toast.success("File uploaded. You can now run containers.", {
      autoClose: 2000,
    });
  };

  const handleRunContainers = async () => {
    if (!uploadedFile) {
      toast.error("Please upload a test file first.", { autoClose: 2000 });
      return;
    }

    if (toRunContainers.length === 0) {
      toast.error("No containers selected to run.", { autoClose: 2000 });
      return;
    }

    setRunning(true);

    for (const container of toRunContainers) {
      const formData = new FormData();
      formData.append("image_name", `user_${container.name}`);
      formData.append("test_file", uploadedFile);

      try {
        const token = user?.token;
        const response = await axios.post(
          "http://127.0.0.1:8000/model/run-container/",
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
              Authorization: `Token ${token}`,
            },
          }
        );

        if (response.status === 200) {
          setResults((prev) => [
            ...prev,
            {
              name: container.name,
              output: response.data.results_file,
              video: response.data.video_file,
            },
          ]);

          // Show video in modal if available
          if (response.data.video_file) {
            const videoURL = `http://127.0.0.1:8000${response.data.video_file}`;
            console.log(videoURL);
            setModalTitle(container.name);
            setModalData(videoURL);
          }

          toast.success(`Run successful for ${container.name}`, {
            autoClose: 2000,
          });
        } else {
          toast.error(`Unexpected response for ${container.name}`, {
            autoClose: 2000,
          });
        }
      } catch (err) {
        console.error(err);
        toast.error(
          err.response?.data?.error || `Failed for ${container.name}`,
          { autoClose: 2000 }
        );
      }
    }

    setRunning(false);
  };

  return (
    <div className="w-screen h-screen bg-[#EAECFF]">
      <AdminNavbar />

      <div className="flex w-full justify-center items-center">
        <div className="bg-white w-[80%] h-[70vh] rounded-2xl mt-20 p-6">
          <div className="flex flex-col h-full justify-between">
            <div className="flex-grow flex justify-between space-x-6">
              {/* Left: Selected Containers */}
              <div className="w-1/3 flex flex-col items-center">
                <div className="bg-green-600 rounded-[15px] p-4 text-white flex flex-col w-[70%] h-[60%]">
                  <h2 className="text-xl font-bold mb-4 text-center">
                    Selected Containers
                  </h2>
                  <div className="overflow-y-auto flex-grow">
                    {selectedContainers.map((container, index) => (
                      <div
                        key={index}
                        className="flex items-center mb-2 justify-center"
                      >
                        <input
                          type="checkbox"
                          id={`container-${index}`}
                          checked={container.toRun}
                          onChange={() => toggleContainer(container)}
                          className="mr-2"
                        />
                        <label
                          htmlFor={`container-${index}`}
                          className="text-center"
                        >
                          {container.name}
                        </label>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Right: Upload + Run */}
              <div className="w-2/3 flex">
                <div className="w-1/2 flex flex-col justify-around items-center">
                  {/* File Upload */}
                  <label
                    className={`${
                      fileUploading ? "bg-gray-400" : "bg-green-600"
                    } text-white py-2 px-4 rounded-full w-[70%] text-center cursor-pointer`}
                  >
                    {fileUploading ? "Uploading..." : "Upload Test File"}
                    <input
                      type="file"
                      className="hidden"
                      disabled={fileUploading}
                      onChange={(e) => {
                        const file = e.target.files[0];
                        if (file) handleFileUpload(file);
                      }}
                    />
                  </label>
                </div>
                <div className="w-1/2 flex flex-col justify-around items-center">
                  <button
                    className={`${
                      running ? "bg-gray-400" : "bg-green-600"
                    } text-white py-2 px-4 rounded-full w-[70%] text-center`}
                    onClick={handleRunContainers}
                    disabled={running}
                  >
                    {running ? "Running..." : "Run Instance"}
                  </button>
                </div>
              </div>
            </div>

            {/* Results */}
            {results.length > 0 && (
              <div className="mt-6 bg-gray-100 rounded-lg p-4 max-h-[30vh] overflow-y-auto">
                <h3 className="font-bold mb-2">Run Results</h3>
                {results.map((r, i) => (
                  <div key={i} className="mb-2">
                    <p className="font-semibold">{r.name}</p>
                    <p className="text-sm">
                      Output File:{" "}
                      {r.output ? (
                        <a
                          href={`http://127.0.0.1:8000${r.output}`}
                          target="_blank"
                          rel="noreferrer"
                          className="text-blue-600 underline"
                        >
                          {r.output}
                        </a>
                      ) : (
                        "Not found"
                      )}
                    </p>
                  </div>
                ))}
              </div>
            )}

            {/* Back Button */}
            <div className="flex justify-end mt-4">
              <button
                className="bg-green-600 text-white py-2 px-10 rounded-full"
                onClick={goBack}
              >
                Back
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Video Modal */}
      {modalData && (
        <div className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-xl max-h-[90vh] overflow-y-auto w-[80vw] flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">{modalTitle} Output</h2>
              <button
                className="bg-red-500 text-white w-8 h-8 rounded-full hover:bg-red-600"
                onClick={() => setModalData(null)}
              >
                ×
              </button>
            </div>
            <div className="flex justify-center">
              <video
                src={modalData}
                controls
                autoPlay
                className="max-w-full max-h-[70vh] rounded"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ContainerTestRun;