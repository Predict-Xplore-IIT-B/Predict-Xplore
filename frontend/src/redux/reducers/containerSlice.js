import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  containers: [],
};

const containerSlice = createSlice({
  name: "containers",
  initialState,
  reducers: {
    // Add list of containers (payload = array of container objects)
    setContainers: (state, action) => {
      state.containers = action.payload.map(c => ({
        id: c.id,
        name: c.name,
        description: c.description,
        created_at: c.created_at,
        selected: false,
        toRun: false,
      }));
    },

    // Toggle selection for a container by id
    toggleContainerSelection: (state, action) => {
      const container = state.containers.find(c => c.id === action.payload);
      if (container) {
        container.selected = !container.selected;
      }
    },

    // Toggle run flag for a container by id
    toggleContainerToRun: (state, action) => {
      const container = state.containers.find(c => c.id === action.payload);
      if (container) {
        container.toRun = !container.toRun;
      }
    },

    // Clear all containers
    clearContainers: (state) => {
      state.containers = [];
    },
  },
});

export const {
  setContainers,
  toggleContainerSelection,
  toggleContainerToRun,
  clearContainers,
} = containerSlice.actions;

export default containerSlice.reducer;