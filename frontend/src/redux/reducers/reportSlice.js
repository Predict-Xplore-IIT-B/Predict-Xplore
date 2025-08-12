import { createSlice } from "@reduxjs/toolkit";

const initialState = {
    reports: [],
    loading: false,
    error: null,
};

const reportSlice = createSlice({
    name: "report",
    initialState,
    reducers: {
        addReport: (state, action) => {
            // Check if report already exists to avoid duplicates
            const existingReport = state.reports.find(report => report.id === action.payload.id);
            if (!existingReport) {
                state.reports.push(action.payload);
            }
        },
        removeReport: (state, action) => {
            state.reports = state.reports.filter(report => report.id !== action.payload);
        },
        clearReports: (state) => {
            state.reports = [];
        },
        setLoading: (state, action) => {
            state.loading = action.payload;
        },
        setError: (state, action) => {
            state.error = action.payload;
            state.loading = false;
        },
    },
});

export const { 
    addReport, 
    removeReport, 
    clearReports, 
    setLoading, 
    setError 
} = reportSlice.actions;

export default reportSlice.reducer;
