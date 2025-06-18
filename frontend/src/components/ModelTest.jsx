import React from 'react'
import ModelTestSelection from './ModelTestSelection'
import { useState } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { useEffect } from 'react'
import { useNavigate } from "react-router-dom";
import { addModelList, updateModelSelection } from '../redux/reducers/modelSlice'
import axios from 'axios'

function ModelTest() {
    const navigate = useNavigate(); 
    const dispatch = useDispatch();
    const user = useSelector((state) => state.user.users[state.user.users.length -1]);
    const [models, setModels] = useState([]);


    // Retriveing list of models from model/list endpoint
    const retrieveData = async() => {
        console.log("Retrieving data");
        try{
            const response = await axios.get('http://localhost:8000/model/list/');
            setModels(response.data.models);
            response.data.models.map((i, index) => {
                dispatch(addModelList(i))
            });
        }
        catch(e){
            console.log("Something went wrong",e)
        }
    }

    useEffect(() => {
        retrieveData();
    }, []);



    // to navigate to ModelTestRun or ModelTestSelection page
    const passdata = () => {
        console.log("user",user);
        
        if (user.role != null){
            navigate("/admin/model-proceed"); // Navigate to the ModelTestRun page
        }
        else{
            navigate("/model-test-run")
        }
    };
    


    return (
    <div className=" bg-[#EAECFF]   flex flex-col items-center justify-between  ">

        {/* Model Selection Section  */}
        <div className="h-[100%] w-[94vw] bg-white rounded-3xl mt-2 mb-10 flex flex-col justify-between items-center">

            {/* Model Selection Title  */}
            <div>
                <p className="text-[#39407D] text-3xl font-medium mt-4">Select Models for Testing</p>
            </div>

            {/* Model Selection Horizontal Scrolling List  */}
            <div className="model-test flex mx-2 w-[95%] p-6 justify-center flex-wrap">
                {models.map((i, index)=>(
                    <div key={i.id}>
                        <ModelTestSelection id = {i.id} name = {i.name} image = {i.model_image} desc = {i.description} index = {index}/>
                    </div>
                ))}
                
            </div>

            {/* Proceed button  */}
            <div className="self-end mr-7">
            <button 
                className="rounded-3xl bg-[#6966FF] text-white px-7 py-2 text-base font-normal mb-3" 
                onClick={passdata}>
                PROCEED
            </button>
        </div>
        </div>
    </div>
  )
}

export default ModelTest