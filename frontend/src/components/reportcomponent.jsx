import '../styles/reportcomponent.css';

export default function ReportComponent(props) {
    console.log(props)
    return (
        <div className="card">
            <div className="model-head">
                <span>{props.model_name}</span>
                <button>Download Report</button>
            </div>
            <div className="model-desc">
                <img src={props.model_img}/>
                <span>{props.description}</span>
            </div>
        </div>
    );
}