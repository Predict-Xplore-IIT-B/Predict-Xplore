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

                if (reports && Array.isArray(reports)) {
                    setReportsData(reports);
                    reports.forEach((report) => dispatch(addReport(report)));
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
        // TODO: Implement bulk download functionality
        console.log("Downloading all reports...");
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
                            model_name={extractModelName(report.report_file)}
                            model_type={extractModelType(report.report_file)}
                            model_bullete={extractModelType(report.report_file)}
                            model_img={modelImage}
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
