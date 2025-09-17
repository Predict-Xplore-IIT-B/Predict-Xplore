import React, { useState } from "react";
import { useSelector } from "react-redux";
import axios from "axios";
import { toast } from "react-toastify";

function ContainerTestRun() {
  const containers = useSelector((state) => state.container.containers);
  const user = useSelector((state) => state.user.users[state.user.users.length -1]);

  const [selectedContainer, setSelectedContainer] = useState("");
  const [testFile, setTestFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleRun = async () => {
    if (!selectedContainer || !testFile) {
      toast.error("Please select a container and upload a test file.");
      return;
    }

    const formData = new FormData();
    formData.append("image_name", selectedContainer);
    formData.append("test_file", testFile);

    try {
      setLoading(true);
      const response = await axios.post(
        "http://localhost:8000/model/run-container/",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
            "Authorization": `Token ${user.token}`,
          },
        }
      );
      console.log("Run result:", response.data);
      setResults(response.data);
      toast.success("Inference complete!");
    } catch (error) {
      console.error("Error running container:", error);
      toast.error("Failed to run container");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-[#EAECFF] min-h-screen flex flex-col items-center p-6">
      <div className="bg-white w-[80%] rounded-3xl p-8 shadow-lg">
        <h2 className="text-3xl font-semibold text-[#39407D] mb-6 text-center">
          Run Container Test
        </h2>

        {/* Select Container */}
        <div className="mb-6">
          <label className="block text-lg font-medium mb-2">Select Container</label>
          <select
            className="w-full border rounded-lg p-3"
            value={selectedContainer}
            onChange={(e) => setSelectedContainer(e.target.value)}
          >
            <option value="">-- Choose Container --</option>
            {containers.map((c) => (
              <option key={c.id} value={c.name}>
                {c.name}
              </option>
            ))}
          </select>
        </div>

        {/* Upload Test File */}
        <div className="mb-6">
          <label className="block text-lg font-medium mb-2">Upload Test File</label>
          <input
            type="file"
            onChange={(e) => setTestFile(e.target.files[0])}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4
                       file:rounded-full file:border-0
                       file:text-sm file:font-semibold
                       file:bg-[#6966FF] file:text-white
                       hover:file:bg-blue-700"
          />
        </div>

        {/* Run Button */}
        <div className="flex justify-center">
          <button
            onClick={handleRun}
            disabled={loading}
            className="bg-green-600 hover:bg-green-700 text-white px-8 py-3 rounded-full text-lg font-bold"
          >
            {loading ? "Running..." : "Run Inference"}
          </button>
        </div>

        {/* Results */}
        {results && (
          <div className="mt-8 p-4 bg-gray-100 rounded-lg">
            <h3 className="text-xl font-semibold text-gray-700 mb-2">Results</h3>
            <p><strong>Detail:</strong> {results.detail}</p>
            {results.results_file && (
              <p>
                <strong>Results File:</strong>{" "}
                <a
                  href={`http://localhost:8000${results.results_file}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 underline"
                >
                  Download
                </a>
              </p>
            )}
            {results.error && (
              <p className="text-red-600"><strong>Error:</strong> {results.error}</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default ContainerTestRun;
