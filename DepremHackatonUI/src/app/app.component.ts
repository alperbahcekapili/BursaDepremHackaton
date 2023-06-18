import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'DepremHackatonUI';
  image = "/assets/Original.png"
  overlay = "/assets/Original.png"
  visualOptions=["Original", "Detections",  "RoadMask","RoadMask Overlay", "Optimal Route", "Virtual Map"]

  overlay_opacity = 0


  changeImage(image_type:string){
    this.image = "/assets/"+image_type+".png";
    if (image_type == "RoadMask Overlay"){
      this.overlay = "/assets/"+"MaskOverlay"+".png";
      this.overlay_opacity = 0.2
    }else{
      this.overlay_opacity = 0
    }
  }


}


/*



class, x1, y1, x2, y2
class ...


QGIS --> Dünya da tam yerini açar  
*/
