import React, { useState, useEffect } from 'react'
import ModelTestSelection from './ModelTestSelection'
import ContainerSelection from './ContainerSelection'
import { useDispatch, useSelector } from 'react-redux'
import { useNavigate } from "react-router-dom";
import { addModelList } from '../redux/reducers/modelSlice'
import { setContainers } from '../redux/reducers/containerSlice'
import axios from 'axios'

function ModelTest() {
  const navigate = useNavigate(); 
  const dispatch = useDispatch();
  const user = useSelector((state) => state.user.users[state.user.users.length -1]);
  const [models, setModels] = useState([]);
  const [containers, setContainersList] = useState([]);

  // Retrieve models
  const retrieveModels = async () => {
    try {
      const response = await axios.get('http://localhost:8000/model/list/');
      setModels(response.data.models);
      response.data.models.forEach((i) => dispatch(addModelList(i)));
    } catch (e) {
      console.log("Error fetching models", e);
    }
  };

  // Retrieve containers
  const retrieveContainers = async () => {
    try {
      const response = await axios.get('http://localhost:8000/model/list-container/');
      setContainersList(response.data.containers);
      dispatch(setContainers(response.data.containers));
    } catch (e) {
      console.log("Error fetching containers", e);
    }
  };

  useEffect(() => {
    retrieveModels();
    retrieveContainers();
  }, []);

  // Proceed with Models
  const proceedModels = () => {
    if (user?.role != null) {
      navigate("/admin/model-proceed");
    } else {
      navigate("/model-test-run");
    }
  };

  // Proceed with Containers
  const proceedContainers = () => {
    if (user?.role != null) {
      navigate("/admin/container-proceed");
    } else {
      navigate("/container-test-run");
    }
  };

  return (
    <div className="bg-[#EAECFF] flex flex-col items-center justify-between">
      
      {/* -------- Models Section -------- */}
      <div className="h-[100%] w-[94vw] bg-white rounded-3xl mt-6 mb-10 flex flex-col justify-between items-center">
        <div>
          <p className="text-[#39407D] text-3xl font-medium mt-4">
            Select Models for Testing
          </p>
        </div>

        <div className="flex mx-2 w-[95%] p-6 justify-center flex-wrap">
          {models.map((m, index) => (
            <div key={m.id}>
              <ModelTestSelection 
                id={m.id} 
                name={m.name} 
                image={m.model_thumbnail} 
                desc={m.description} 
                index={index}
              />
            </div>
          ))}
        </div>

        <div className="self-end mr-7">
          <button 
            className="rounded-3xl bg-[#6966FF] text-white px-7 py-2 text-base font-normal mb-3"
            onClick={proceedModels}
          >
            PROCEED MODELS
          </button>
        </div>
      </div>

      {/* -------- Containers Section -------- */}
      <div className="h-[100%] w-[94vw] bg-white rounded-3xl mt-6 mb-10 flex flex-col justify-between items-center">
        <div>
          <p className="text-[#39407D] text-3xl font-medium mt-4">
            Select Containers for Testing
          </p>
        </div>

        <div className="flex mx-2 w-[95%] p-6 justify-center flex-wrap">
          {containers.map((c, index) => (
            <div key={c.id}>
              <ContainerSelection 
                id={c.id} 
                name={c.name} 
                image={c.thumbnail || null} 
                desc={c.description} 
                index={index}
              />
            </div>
          ))}
        </div>

        <div className="self-end mr-7">
          <button 
            className="rounded-3xl bg-green-600 hover:bg-green-700 text-white px-7 py-2 text-base font-normal mb-3"
            onClick={proceedContainers}
          >
            PROCEED CONTAINERS
          </button>
        </div>
      </div>

    </div>
  )
}

export default ModelTest
