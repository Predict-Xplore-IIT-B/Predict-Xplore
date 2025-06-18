import '../styles/reportcomponent.css';

export default function ReportComponent(props) {
    const handleDownload = async () => {
    const filename = `${props.username}_${props.model_name}`;
    const url = `http://localhost:8000/model/download/report/${encodeURIComponent(filename)}`;

    try {
        const response = await fetch(url);
        if (response.ok) {
            window.open(url, '_blank');
        } else {
            alert("Report not found.");
        }
    } catch (err) {
        console.error("Error fetching report:", err);
        alert("Something went wrong while downloading the report.");
    }
};


    return (
        <div className="card">
            <div className="model-head">
                <span>{props.model_name}</span>
                <button onClick={handleDownload}>Download Report</button>
            </div>
            <div className="model-desc">
                <img src={props.model_img} alt={props.model_name} />
                <span>{props.description}</span>
            </div>
        </div>
    );
}
