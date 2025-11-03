import React, { useState } from "react";
import { useDispatch } from "react-redux";
import { toggleContainerSelection } from "../redux/reducers/containerSlice";
import "../styles/modeltestselection.css"; // reuse same css

function ContainerSelection(props) {
  const dispatch = useDispatch();
  const [select, setSelect] = useState(props.selected || false);

  const handleSelectToggle = () => {
    dispatch(toggleContainerSelection(props.id));
    setSelect(!select);
  };

  return (
    <div className="card h-[50vh] w-[18vw] rounded-3xl shadow-custom m-6 min-w-[17rem] flex flex-col items-center justify-between min-h-[20rem]">
      <div className="card-inner h-[85%]">
        {/* Front */}
        <div className="card-front flex flex-col items-center h-[100%] m-1">
          <div className="title text-[#39407D] text-xl font-medium text-center">
            <p>{props.name}</p>
          </div>
          {props.image && (
            <div className="mb-1 object-contain">
              <img
                className="rounded-xl max-h-[30vh]"
                src={`http://localhost:8000/media/${props.image}`}
                alt="No Image"
              />
            </div>
          )}
        </div>

        {/* Description */}
        <div className="card-back h-[100%] overflow-y-auto m-1 p-1 text-justify">
          <p>{props.desc}</p>
        </div>
      </div>

      {/* Selection Button */}
      <div>
        <button
          className={`test_select_btn ${select ? "selected" : "not_selected"}`}
          onClick={handleSelectToggle}
        >
          {select ? "Selected" : "Not Selected"}
        </button>
      </div>
    </div>
  );
}

export default ContainerSelection;
