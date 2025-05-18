'use client';
import { useState } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { addProvider } from '@/lib/api';

export default function AddProviderForm() {
  const [alias, setAlias] = useState('');
  const [modelName, setModelName] = useState('');
  const [envVar, setEnvVar] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setMessage(null);
    try {
      await addProvider({ alias, model_name: modelName, env_var_name: envVar, api_key: apiKey });
      setMessage('Provider added successfully');
      setAlias('');
      setModelName('');
      setEnvVar('');
      setApiKey('');
    } catch (err) {
      console.error(err);
      setMessage('Failed to add provider');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4 max-w-md">
      <div>
        <Label htmlFor="alias">Model Alias</Label>
        <Input id="alias" value={alias} onChange={e => setAlias(e.target.value)} required />
      </div>
      <div>
        <Label htmlFor="model">Model Name</Label>
        <Input id="model" value={modelName} onChange={e => setModelName(e.target.value)} required />
      </div>
      <div>
        <Label htmlFor="env">ENV Variable Name</Label>
        <Input id="env" value={envVar} onChange={e => setEnvVar(e.target.value)} required />
      </div>
      <div>
        <Label htmlFor="key">API Key</Label>
        <Input id="key" type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} required />
      </div>
      {message && <p className="text-sm">{message}</p>}
      <Button type="submit" disabled={loading}>{loading ? 'Saving...' : 'Add Provider'}</Button>
    </form>
  );
}
