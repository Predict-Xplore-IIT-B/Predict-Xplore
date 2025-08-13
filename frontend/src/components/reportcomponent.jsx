import '../styles/reportcomponent.css';

export default function ReportComponent(props) {
    const handleDownload = async () => {
        // Add trailing slash to match Django URL pattern
        const url = `http://localhost:8000/model/download/report/${props.report_id}/`;
        
        console.log("Downloading report from:", url);

        try {
            const response = await fetch(url, {
                headers: {
                    'Authorization': `Bearer ${props.userToken}` // Add auth if needed
                }
            });
            
            if (response.ok) {
                // Create a blob from the response
                const blob = await response.blob();
                const downloadUrl = window.URL.createObjectURL(blob);
                
                // Format datetime as YYYYMMDDHHMMSS
                const now = new Date();
                const datetime = now.getFullYear().toString() + 
                    (now.getMonth() + 1).toString().padStart(2, '0') + 
                    now.getDate().toString().padStart(2, '0') + 
                    now.getHours().toString().padStart(2, '0') + 
                    now.getMinutes().toString().padStart(2, '0') + 
                    now.getSeconds().toString().padStart(2, '0');
                
                // Format filename: username_modeltype_datetime.pdf
                const modelType = props.model_type.toLowerCase().replace(/\s+/g, '_');
                const filename = `${props.username}_${modelType}_${datetime}.pdf`;
                
                // Create a temporary link element to trigger download
                const link = document.createElement('a');
                link.href = downloadUrl;
                link.download = filename;
                document.body.appendChild(link);
                link.click();
                
                // Clean up
                document.body.removeChild(link);
                window.URL.revokeObjectURL(downloadUrl);
            } else {
                console.error(`HTTP Error: ${response.status} ${response.statusText}`);
                alert("Report not found.");
            }
        } catch (err) {
            console.error("Error downloading report:", err);
            alert("Something went wrong while downloading the report.");
        }
    };

    const handleView = () => {
        const url = `http://localhost:8000/media/${props.report_file}`;
        window.open(url, "_blank", "noopener,noreferrer");
    };

    return (
        <div className="card">
            <div className="model-head">
                <div className="model-info">
                    <span className="model-name">{props.model_name}</span>
                    <span className="model-type">
                    Model Type: <span style={{ color: "#6966FF" }}>{props.model_type}</span>
                    </span>
                    {/* <span className="report-date">
                        Created: {new Date(props.created_at).toLocaleDateString()}
                    </span> */}
                </div>
                <div className="button-column">
                    <button className="report-btn" onClick={handleDownload}>Download Report</button>
                    <button className="report-btn" onClick={handleView}>View Report</button>
                </div>

            </div>
            <div className="model-desc">
                <img className='rounded-xl max-h-[30vh]' src={props.model_img} alt={props.model_name} />
                <div className="report-details">
                    <p><strong>Test Case ID:</strong> {props.test_case_id}</p>
                    <p><strong>Report ID:</strong> {props.report_id}</p>
                    <p className="description">{props.description}</p>
                </div>
            </div>
        </div>
    );
}
