#Features:

- implement auto-selection of QLoRA
Hardware Tier Detection and Optimization
The system now automatically detects your GPU's VRAM capacity and assigns it to one of these tiers:

High-End Tier (16GB+ VRAM) - For RTX 3090, RTX 4090, etc.
Adapter Type: QLoRA
Quantization: 8-bit (better quality with acceptable memory cost)
Rank: 16 (higher for better adaptation)
Alpha: 32 (more expressive updates)
Target Modules: More comprehensive (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)

Medium Tier (8-16GB VRAM) - For RTX 3070, RTX 3080, RTX 2080 SUPER, etc.
Adapter Type: QLoRA
Quantization: 4-bit (efficient memory usage)
Rank: 12 (balanced)
Alpha: 24 (balanced)
    Target Modules: Moderate coverage (q_proj, k_proj, v_proj, o_proj)

Low Tier (4-8GB VRAM) - For GTX 1650, RTX 2060, etc.
Adapter Type: QLoRA
Quantization: 4-bit (maximum memory efficiency)
Rank: 8 (memory-efficient)
Alpha: 16 (proportional)
    Target Modules: Limited (q_proj, v_proj)

Minimal Tier (<4GB VRAM or CPU-only)
Adapter Type: Standard LoRA (no quantization)
Rank: 4 (minimal)
Alpha: 8 (minimal)
Target Modules: Minimal (q_proj only)

# Getting Started with Create React App

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you're on your own.

You don't have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

### Code Splitting

This section has moved here: [https://facebook.github.io/create-react-app/docs/code-splitting](https://facebook.github.io/create-react-app/docs/code-splitting)

### Analyzing the Bundle Size

This section has moved here: [https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size](https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size)

### Making a Progressive Web App

This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)

### Advanced Configuration

This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)

### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

### `npm run build` fails to minify

This section has moved here: [https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify](https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify)
