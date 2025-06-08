import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import SelectCheckbox from "../../components/SelectCheckbox";
import AdminNavbar from "../../components/AdminNavbar";
import { useSelector, useDispatch } from "react-redux";
import { toggleModelToRun } from "../../redux/reducers/modelSlice";
import axios from "axios";
import { toast } from "react-toastify";

function ModelProceed() {
  const navigate = useNavigate();
  const dispatch = useDispatch();

  const user = useSelector((state) => state.user.users[state.user.users.length - 1]);
  const selectedModels = useSelector((state) =>
    state.models.modelArray.filter((model) => model.selected)
  );
  const toRunModels = useSelector((state) =>
    state.models.modelArray.filter((model) => model.toRun)
  );

  const [imageSrcs, setImageSrcs] = useState([]);
  const [imgUpload, setImgUpload] = useState(false);
  const [runInstance, setRunInstance] = useState(false);

  const allClasses = ["class 1", "class 2", "class 3", "class 4"];
  const algorithms = ["Algorithm 1", "Algorithm 2", "Algorithm 3", "Algorithm 4"];

  const toggleModel = (model) => {
    dispatch(toggleModelToRun(model.id));
  };

  const goBack = () => {
    const route = user.role === "admin" ? "/admin/model-test" : "/model-test";
    navigate(route);
  };

  const uploadTestImage = async (file) => {
    const formData = new FormData();
    formData.append("image", file);
    setImgUpload(true);

    try {
      const imgResponse = await axios.post(
        "http://127.0.0.1:8000/model/instance/upload",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      if (imgResponse.status === 200) {
        toast.success("Image uploaded successfully", { autoClose: 2000 });
      }
    } catch (e) {
      toast.error("Error in uploading image", { autoClose: 2000 });
    } finally {
      setImgUpload(false);
    }
  };

  const handleRunInstance = async () => {
    setRunInstance(true);
    setImageSrcs([]);

    for (const model of toRunModels) {
      const data = {
        username: user.username,
        models: [model.name],
      };

      try {
        const response = await axios.post(
          "http://127.0.0.1:8000/model/instance/predict",
          data,
          { headers: { "Content-Type": "application/json" } }
        );

        if (response.status === 200) {
          try {
            const imageResponse = await axios.get(
              `http://127.0.0.1:8000/model/output/${user.username}/${model.name}`,
              { responseType: "blob" }
            );
            const imageURL = URL.createObjectURL(imageResponse.data);
            setImageSrcs((prev) => [...prev, { name: model.name, url: imageURL }]);
          } catch (error) {
            console.error(`Error fetching image for model ${model.name}:`, error);
          }
        }
      } catch (error) {
        console.error("Prediction error:", error);
        toast.error(
          error.response?.data?.error || `Failed for model: ${model.name}`,
          { autoClose: 2000 }
        );
      }
    }

    setRunInstance(false);
  };

  const closeImageView = () => {
    setImageSrcs([]);
  };

  return (
    <div className="w-screen h-screen bg-[#EAECFF]">
      <AdminNavbar />

      {/* Image Overlay Gallery */}
      {imageSrcs.length > 0 && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-60 z-50">
          <div className="relative bg-white p-6 rounded-xl max-h-[90vh] overflow-y-auto">
            <button
              onClick={closeImageView}
              className="absolute top-2 right-2 bg-red-500 text-white w-8 h-8 rounded-full hover:bg-red-600"
            >
              ×
            </button>
            <div className="flex space-x-6 overflow-x-auto max-w-[90vw] py-4 px-2">
              {imageSrcs.map((img, index) => (
                <div key={index} className="flex-shrink-0 flex flex-col items-center">
                  <p className="mb-2 font-semibold text-center">{img.name}</p>
                  <img
                    src={img.url}
                    alt={`Output for ${img.name}`}
                    className="w-[300px] h-[300px] object-contain rounded shadow"
                  />
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex w-full justify-center items-center">
        <div className="bg-white w-[80%] h-[70vh] rounded-2xl mt-20">
          <div className="flex flex-col h-full justify-center">
            <div className="flex-grow space-x-6 flex mb-6 justify-between">
              {/* Selected Models */}
              <div className="w-1/3 flex flex-col items-center justify-center">
                <div className="bg-[#6966FF] rounded-[15px] p-4 text-white flex flex-col w-[70%] h-[60%]">
                  <h2 className="text-xl font-bold mb-4 text-center">
                    Selected Models
                  </h2>
                  <div className="overflow-y-auto flex-grow">
                    {selectedModels.map((model, index) => (
                      <div
                        key={index}
                        className="flex items-center mb-2 justify-center"
                      >
                        <input
                          type="checkbox"
                          id={`model-${index}`}
                          checked={model.toRun}
                          onChange={() => toggleModel(model)}
                          className="custom-checkbox mr-2"
                        />
                        <label htmlFor={`model-${index}`} className="text-center">
                          {model.name}
                        </label>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Right Side */}
              <div className="w-2/3 flex">
                <div className="w-1/2 flex flex-col justify-around items-center">
                  <SelectCheckbox options={allClasses} title="Select Class" />
                  <label className={`${imgUpload ? "bg-gray-400" : "bg-[#6966FF]"} text-white py-2 px-4 rounded-full w-[70%] text-center cursor-pointer`}>
                    {imgUpload ? "Image Uploading ..." : "Upload Test Image"}
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
                </div>
                <div className="w-1/2 flex flex-col justify-around items-center">
                  <SelectCheckbox options={algorithms} title="Select XAI Algo" />
                  <button
                    className={`${runInstance ? "bg-gray-400" : "bg-[#6966FF]"} text-white py-2 px-4 rounded-full w-[70%]`}
                    onClick={handleRunInstance}
                    disabled={runInstance}
                  >
                    {runInstance ? "Running..." : "Run Instance"}
                  </button>
                </div>
              </div>
            </div>

            {/* Back Button */}
            <div className="flex justify-end mb-5 mr-5">
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
    </div>
  );
}

export default ModelProceed;
