import { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import ReportComponent from './reportcomponent.jsx';
import '../styles/Report.css';
import axios from 'axios';
import { addReport } from '../redux/reducers/reportSlice';
import modelImage from '../assets/model.img.png';

export default function Report() {
    const [reportsData, setReportsData] = useState([]);
    const [loading, setLoading] = useState(false);
    const dispatch = useDispatch();
    const user = useSelector((state) => state.user.users[state.user.users.length - 1]);

    useEffect(() => {
        const fetchReportsData = async () => {
            setLoading(true);
            try {
                const response = await axios.get('http://localhost:8000/model/report/');               
                const reports = response.data.reports;
                console.log("reports:",reports);
                
                if (reports && Array.isArray(reports)) {
                     // Sort newest to oldest
                    const sortedReports = [...reports].sort((a, b) => 
                        new Date(b.created_at) - new Date(a.created_at)
                    );
                    setReportsData(sortedReports);
                    sortedReports.forEach((report) => dispatch(addReport(report)));
                } else {
                    console.error("Unexpected response structure:", response.data);
                }
            } catch (error) {
                console.error("Error fetching reports:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchReportsData();
    }, [dispatch]);

    const downloadAllReports = async () => {
        if (!reportsData.length) {
            alert("No reports to download.");
            return;
        }

        for (const report of reportsData) {
            const url = report.report_file_url; // assuming API gives this
            try {
                const response = await fetch(url);
                if (response.ok) {
                    const blob = await response.blob();
                    const link = document.createElement('a');
                    link.href = URL.createObjectURL(blob);
                    link.download = url.split('/').pop(); // use filename from URL
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                } else {
                    console.error(`Failed to download ${url}`);
                }
            } catch (err) {
                console.error(`Error downloading ${url}:`, err);
            }
        }
    };

    // Extract model name from report file name
    const extractModelName = (fileName) => {
        const parts = fileName.split('_');
        if (parts.length >= 3) {
            return parts.slice(1, -1).join(' ').replace(/\.(pdf|json)$/, '');
        }
        return 'Unknown Model';
    };

    // Extract model type from report file name
    const extractModelType = (fileName) => {
        if (fileName.includes('human_detection')) return 'Human Detection';
        if (fileName.includes('image_segmentation')) return 'Image Segmentation';
        if (fileName.includes('object_detection')) return 'Object Detection';
        return 'Unknown Type';
    };

    if (loading) {
        return (
            <div className="Page">
                <div className="loading-container">
                    <p>Loading reports...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="Page">
            <button className="Download" onClick={downloadAllReports}>
                Download Complete Report
            </button>
            <div className="cardHolder">
                {reportsData.length > 0 ? (
                    reportsData.map((report) => (
                        <ReportComponent
                        key={report.id}
                        report_id={report.id}
                        test_case_id={report.test_case__id}
                        model_name={report.model__name || extractModelName(report.report_file)}
                        model_type={
                            report.model__model_type
                            ? ({
                                HumanDetection: 'Human Detection',
                                ObjectDetection: 'Object Detection',
                                ImageSegmentation: 'Image Segmentation',
                                }[report.model__model_type] || 'Unknown Type')
                            : extractModelType(report.report_file)
                        }
                        model_img={report.model__model_thumbnail_url || modelImage}
                        report_file={report.report_file}
                        created_at={report.created_at}
                        description={`Report generated on ${new Date(report.created_at).toLocaleDateString()}`}
                        username={user?.username || 'Guest'}
                        />

                    ))
                ) : (
                    <div className="no-reports">
                        <p>No reports available</p>
                    </div>
                )}
            </div>
        </div>
    );
}
