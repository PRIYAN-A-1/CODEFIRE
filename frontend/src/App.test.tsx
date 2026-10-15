import React from 'react';
import {afterEach,expect,test,vi} from 'vitest';
import {render,screen,cleanup,fireEvent,waitFor} from '@testing-library/react';
import {App} from './main';
afterEach(()=>{cleanup();vi.unstubAllGlobals();});
test('shows backend errors without fabricating model results',async()=>{vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('offline')));render(<App/>);expect(screen.getByText('Build from first principles.')).toBeTruthy();await waitFor(()=>expect(screen.getByRole('alert').textContent).toContain('offline'));});
test('workspace has accessible prompt and real generation control',()=>{vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('offline')));render(<App/>);fireEvent.click(screen.getByRole('button',{name:'Coding workspace'}));expect(screen.getByLabelText('Source prompt')).toBeTruthy();expect(screen.getByRole('button',{name:'Generate continuation'})).toBeTruthy();});
