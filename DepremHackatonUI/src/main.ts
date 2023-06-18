import { platformBrowserDynamic } from '@angular/platform-browser-dynamic';

import { AppModule } from './app/app.module';


platformBrowserDynamic().bootstrapModule(AppModule)
  .catch(err => console.error(err));


  // Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBRWrVcy7JPCB87tSZkGl29olWHTPGL5DA",
  authDomain: "aita-demo.firebaseapp.com",
  projectId: "aita-demo",
  storageBucket: "aita-demo.appspot.com",
  messagingSenderId: "904569706105",
  appId: "1:904569706105:web:e362310db2a059950b66a8"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);


