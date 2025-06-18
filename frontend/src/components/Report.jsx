import { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import ReportComponent from './reportcomponent.jsx';
import '../styles/Report.css';
import axios from 'axios';
import { addModelList } from '../redux/reducers/modelSlice';
import modelImage from '../assets/model.img.png'; // ✅ Import static image

export default function Report() {
    const [modelData, setModelData] = useState([]);
    const dispatch = useDispatch();
    const user = useSelector((state) => state.user.users[state.user.users.length - 1]);

    useEffect(() => {
        const fetchModelData = async () => {
            try {
                const response = await axios.get('http://localhost:8000/model/list/');
                const models = response.data.models;

                if (models && Array.isArray(models)) {
                    setModelData(models);
                    models.forEach((model) => dispatch(addModelList(model)));
                } else {
                    console.error("Unexpected response structure:", response.data);
                }
            } catch (error) {
                console.error("Error fetching model list:", error);
            }
        };

        fetchModelData();
    }, [dispatch]);

    return (
        
            <div className="Page">
                <button className="Download">Download Complete Report</button>
                <div className="cardHolder">
                    {modelData.map((data, index) => (
                        <ReportComponent
                            key={index}
                            model_bullete={data.model_type}
                            model_img={modelImage}
                            model_name={data.name}
                            description={data.description}
                            username={user?.username || 'Guest'}
                        />
                    ))}
                </div>
            </div>
        
    );
}
