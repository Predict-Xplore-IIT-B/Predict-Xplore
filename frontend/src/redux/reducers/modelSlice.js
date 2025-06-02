import { createSlice } from '@reduxjs/toolkit';

// Initial state for the model list with an empty array
const initialState = {
  modelArray: [],
};

const modelSlice = createSlice({
  name: 'models',
  initialState,
  reducers: {
    // 1. Add initial list of models (payload contains model object)
    addModelList: (state, action) => {
      const newModel = {
        id: action.payload.id,
        name: action.payload.name,
        description: action.payload.description,
        model_type: action.payload.model_type,
        created_at: action.payload.created_at,
        selected: false,
        toRun: false,
      };
      state.modelArray.push(newModel);
    },

    // 2. Delete a specific model based on its name
    deleteModelByName: (state, action) => {
      state.modelArray = state.modelArray.filter(
        (model) => model.name !== action.payload
      );
    },

    // 3. Toggle "selected" value of a specific model by name
    updateModelSelection: (state, action) => {
      const model = state.modelArray.find((m) => m.id === action.payload);
      if (model) {
        model.selected = !model.selected;
      }
    },

    // 4. Toggle "toRun" value of a specific model by id
    toggleModelToRun: (state, action) => {
      const model = state.modelArray.find((m) => m.id === action.payload);
      if (model) {
        model.toRun = !model.toRun;
      }
    },

    // 5. Clear the model list
    clearModelList: (state) => {
      state.modelArray = [];
    },
  },
});

// Export actions for use in components
export const {
  addModelList,
  deleteModelByName,
  updateModelSelection,
  toggleModelToRun,
  clearModelList,
} = modelSlice.actions;

// Export reducer to be used in the store
export default modelSlice.reducer;
