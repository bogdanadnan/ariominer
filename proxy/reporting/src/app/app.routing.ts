import { Routes } from '@angular/router';

import { DashboardComponent }   from './dashboard/dashboard.component';
import { WorkerComponent }   from './worker/worker.component';

export const AppRoutes: Routes = [
    {
        path: '',
        redirectTo: 'dashboard',
        pathMatch: 'full',
    },
    {
        path: 'dashboard',
        component: DashboardComponent
    },
    {
        path: 'worker/:id',
        component: WorkerComponent
    }
]
