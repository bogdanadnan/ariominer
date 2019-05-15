import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';
import { NgModule } from '@angular/core';
import { RouterModule } from '@angular/router';

import { AppComponent } from './app.component';
import { AppRoutes } from './app.routing';
import { SidebarModule } from './sidebar/sidebar.module';

import { DashboardComponent }   from './dashboard/dashboard.component';
import { WorkerComponent }   from './worker/worker.component';
import {SidebarService} from "./sidebar/sidebar.service";
import {DashboardService} from "./dashboard/dashboard.service";
import {WorkerService} from "./worker/worker.service";

@NgModule({
  declarations: [
    AppComponent,
    DashboardComponent,
    WorkerComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    RouterModule.forRoot(AppRoutes),
    SidebarModule
  ],
  providers: [
      SidebarService,
      DashboardService,
      WorkerService
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
