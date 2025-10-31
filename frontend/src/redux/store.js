import { configureStore } from "@reduxjs/toolkit";
import userReducer from "./reducers/userSlice";
import modelReducer from "./reducers/modelSlice";
import containerReducer from "./reducers/containerSlice";

export const store = configureStore({
  reducer: {
    user: userReducer, 
    models: modelReducer,
    containers: containerReducer,
  },
});

export default store;