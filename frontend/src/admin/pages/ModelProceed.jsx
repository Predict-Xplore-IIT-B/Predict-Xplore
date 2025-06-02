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
  const selectedModels = useSelector((state) => state.models.modelArray.filter((model) => model.selected));
  const toRunModels = useSelector((state) => state.models.modelArray.filter((model) => model.toRun));

  const [imageSrc, setImageSrc] = useState(null);
  const [imgUpload, setImgUpload] = useState(false);
  const [runInstance, setRunInstance] = useState(false);

  const allClasses = ["class 1", "class 2", "class 3", "class 4"];
  const algorithms = ["Algorithm 1", "Algorithm 2", "Algorithm 3", "Algorithm 4",];

  // to toggle the model to run
  const toggleModel = (model) => {
    dispatch(toggleModelToRun(model.id));
  };

  // to navigate back
  const goBack = () => {
    const route = user.role === "admin" ? "/admin/model-test" : "/model-test";
    navigate(route);
  };

  // to upload test image
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
    }
    finally {
      setImgUpload(false);
    }
  };

  // to handle instance run
  const handleRunInstance = async () => {
    const data = {
      username: user.username,
      models: [toRunModels[0]?.name],
    };

    setRunInstance(true);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/model/instance/predict",
        data,
        { headers: { "Content-Type": "application/json" }}
      );

      if (response.status === 200) {
        // toast.success("Instance run successfully", { autoClose: 2000 });

        try {
          const imageResponse = await axios.get(
            `http://127.0.0.1:8000/model/output/${data.username}/${data.models[0]}`,
            { responseType: "blob" }
          );
          const imageURL = URL.createObjectURL(imageResponse.data);
          setImageSrc(imageURL);
        } catch (error) {
          console.error("Error fetching image:", error);
        }
      }
    } catch (error) {
      console.error("Error Details:", error.response || error.message);
      toast.error(
        error.response?.data?.error || "An error occurred while running the instance.",
        { autoClose: 2000 }
      );
    }
    finally{
      setRunInstance(false);
    }
  };

  // function to close the image view
  const closeImageView = () => {
    setImageSrc(null);
  };

  return (
    <div className="w-screen h-screen bg-[#EAECFF]">
      <AdminNavbar />

      {/* Image Overlay */}
      {imageSrc && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
          <div className="relative">
            {/* Close Button */}
            <button
              onClick={closeImageView}
              className="absolute top-2 right-2 bg-red-500 text-white w-8 h-8 rounded-full flex items-center justify-center hover:bg-red-600 transition"
            >
              ×
            </button>

            {/* Display Image */}
            <img
              src={imageSrc}
              alt="Inference Output"
              className="w-[80%] h-[80%] object-contain"
            />
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex w-[100%] justify-center items-center">
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
                  <label className={`${imgUpload ? "bg-gray-400" : "bg-[#6966FF]"} text-white py-2 px-4 rounded-full flex items-center justify-center w-[70%] mr- cursor-pointer`}>
                    {imgUpload ? "Image Uploading ..." : "Upload Test Image"}
                    <input
                      disabled={imgUpload}
                      type="file"
                      accept=".jpg,.jpeg,.png"
                      className="hidden"
                      onChange={(event) => {
                        const file = event.target.files[0];
                        if (file) {
                          uploadTestImage(file);
                        }
                      }}
                    />
                  </label>
                </div>
                <div className="flex flex-col w-1/2 justify-around items-center">
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
