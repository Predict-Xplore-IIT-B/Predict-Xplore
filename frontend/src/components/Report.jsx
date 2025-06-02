import modelData from '../data/reportdata.jsx';
import ReportComponent from './reportcomponent.jsx';
import '../styles/Report.css'
import Navbar from './Navbar.jsx';

export default function Report(){
    return(
        <>
            <div className="Page">
                <button className='Download'>Download Complete Report</button>
                <div className='cardHolder'>
                    {modelData.map((data,index)=>(
                        <ReportComponent
                        key={index}
                        model_bullete={data.model_bullete}
                        model_img={data.model_img}
                        model_name={data.model_name}
                        description={data.description}
                        />

                    ))}
                </div>
            </div>
        </>
    );
}