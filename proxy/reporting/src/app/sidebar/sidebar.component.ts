import { Component, OnInit } from '@angular/core';
import { SidebarService } from "./sidebar.service";
import {interval} from "rxjs/observable/interval";

declare var $:any;

export interface RouteInfo {
    path: string;
    title: string;
    icon: string;
    class: string;
}

class Worker {
    worker_id : String;
    worker_name : String;
}

@Component({
    moduleId: module.id,
    selector: 'sidebar-cmp',
    templateUrl: 'sidebar.component.html',
})

export class SidebarComponent implements OnInit {
    constructor(private sidebarService: SidebarService) { }

    public menuItems: any[];

    private oneMinuteUpdater : any;

    ngOnInit() {
        this.menuItems = [ { path: "dashboard", title: "Dashboard", icon: 'ti-panel', class: 'active' } ];

        this.oneMinuteUpdater = interval(60000).subscribe(i => {
            this.updateMenuItems();
        });

        this.updateMenuItems();
    }

    updateMenuItems() {
        this.sidebarService.getMenuItems().subscribe((data : Worker[]) => {
            for(var i = this.menuItems.length - 1;i > 0; i--) {
                var old_path = this.menuItems[i].path;
                var found = false;
                for(var j=0;j<data.length;j++) {
                    if(("worker/" + data[j].worker_id) == old_path) {
                        found = true;
                        break;
                    }
                }
                if(!found) {
                    this.menuItems.splice(i, 1);
                }
            }

            for(var i = 0; i<data.length;i++) {
                var new_path = ("worker/" + data[i].worker_id);
                var found = false;
                for(var j=1;j<this.menuItems.length;j++) {
                    if(this.menuItems[j].path == new_path) {
                        found = true;
                        break;
                    }
                }
                if(!found) {
                    this.menuItems.push({
                        path: new_path,
                        title: data[i].worker_name,
                        icon: 'ti-hummer',
                        class: ''
                    });
                }
            }
        });
    }

    isNotMobileMenu(){
        if($(window).width() > 991){
            return false;
        }
        return true;
    }

}
