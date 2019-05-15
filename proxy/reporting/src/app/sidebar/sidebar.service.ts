import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable()
export class SidebarService {
    constructor(private http: HttpClient) { }

    getMenuItems() {
        return this.http.get("/api?q=getWorkers");
    }
}
