import React, { useState } from 'react'
import "../styles/modeltestselection.css"
import { useDispatch, useSelector } from 'react-redux'
import { addModelList, updateModelSelection } from '../redux/reducers/modelSlice'

function ModelTestSelection(props) {

  const dispatch = useDispatch();
  
  const [select, setSelect] = useState(false)// modelArray[props.index].status

  const handleSelectToggle = () => {
    dispatch(updateModelSelection(props.id))
    setSelect(!select)
  };  


  return (
    <div className="card h-[50vh] w-[18vw] rounded-3xl shadow-custom m-6 min-w-[17rem] flex flex-col items-center justify-between min-h-[20rem] below-1488:h-[23rem]">
      <div className="card-inner h-[85%] ">

        <div className="card-front flex flex-col items-center  h-[100%]  m-1">

          {/* Model title */}
          <div className="title text-[#39407D] text-xl font-medium text-center">
            <p>{props.name}</p>
          </div>

          {/* Model Image */}
          <div className="mb-1 object-contain">
            <img className="rounded-xl max-h-[30vh]" src={props.image} alt="No Image" />
          </div>

        </div>

        {/* Model Description */}
        <div className="card-back h-[100%] overflow-y-auto m-1 p-1 text-justify">
          <p>{props.desc}</p>
        </div>

      </div>

      {/* Model Selection Button */}
      <div>
        <button className={`test_select_btn ${select?'selected':'not_selected'}`} onClick={handleSelectToggle}>{select?"Selected":"Not Selected"}</button>
      </div>
    </div>
  )
}

export default ModelTestSelection